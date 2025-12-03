import os
import sys

# 将项目根目录添加到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

# 现在，当你 import neural_renderer 时，Python 会先在项目目录下寻找

from PIL import Image
import numpy as np
import tqdm
import torch
import cv2
import warnings

warnings.filterwarnings("ignore")

from neural_renderer.load_obj import load_obj

from torchvision.transforms import Resize
from data_loader_new import MyDataset
from torch.utils.data import Dataset, DataLoader
from grad_cam import CAM

import torch.nn.functional as F
import random
from functools import reduce
import argparse

torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)


# 初始化参数与配置
parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=1)
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--batchsize", type=int, default=1)
parser.add_argument("--lamb", type=float, default=1e-4)
parser.add_argument("--d1", type=float, default=0.9)
parser.add_argument("--d2", type=float, default=0.1)
parser.add_argument("--t", type=float, default=0.0001)
parser.add_argument("--obj", type=str, default='audi_et_te.obj')
parser.add_argument("--faces", type=str, default='./all_faces.txt')
parser.add_argument("--datapath", type=str, default="./data")
parser.add_argument("--content", type=str, default="./textures/smile.npy")
parser.add_argument("--canny", type=str, default="./textures/smile_canny.npy")
args = parser.parse_args()

# 从.obj文件中加载 3D 模型的顶点（vertices）、面（faces）和初始纹理（textures）。
obj_file = args.obj
texture_size = 6
vertices, faces, textures = load_obj(filename_obj=obj_file, texture_size=texture_size, load_texture=True)

if args.datapath is None:
    raise ValueError("datapath参数未正确设置，请确保传入有效的数据路径")
mask_dir = os.path.join(args.datapath, 'masks/')

# torch.autograd.set_detect_anomaly(True) # 开启会影响性能，调试时再用

log_dir = ""


def make_log_dir(logs):
    global log_dir
    dir_name = ""
    for key in logs.keys():
        dir_name += str(key) + '-' + str(logs[key]) + '+'
    dir_name = 'logs/' + dir_name.rstrip('+')  # 移除末尾多余的 '+'
    print(dir_name)
    if not (os.path.exists(dir_name)):
        os.makedirs(dir_name)
    log_dir = dir_name


T = args.t
D1 = args.d1
D2 = args.d2
lamb = args.lamb
LR = args.lr
BATCH_SIZE = args.batchsize
EPOCH = args.epoch

texture_content = torch.from_numpy(np.load(args.content)).cuda(device=0)
texture_canny = torch.from_numpy(np.load(args.canny)).cuda(device=0)
texture_canny = (texture_canny >= 1).int()


def loss_content_diff(tex):
    return D1 * torch.sum(texture_canny * torch.pow(tex - texture_content, 2)) + D2 * torch.sum(
        (1 - texture_canny) * torch.pow(tex - texture_content, 2))


def loss_smooth(img, mask):
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    mask = mask[:, :-1, :-1]
    mask = mask.unsqueeze(1)
    return T * torch.sum(mask * (s1 + s2))


cam_edge = 7
vis = np.zeros((cam_edge, cam_edge))


def dfs(x1, x, y, points):
    points.append(x1[x][y])
    global vis
    vis[x][y] = 1
    n = 1
    if x + 1 < cam_edge and x1[x + 1][y] > 0 and not vis[x + 1][y]:
        n += dfs(x1, x + 1, y, points)
    if x - 1 >= 0 and x1[x - 1][y] > 0 and not vis[x - 1][y]:
        n += dfs(x1, x - 1, y, points)
    if y + 1 < cam_edge and x1[x][y + 1] > 0 and not vis[x][y + 1]:
        n += dfs(x1, x, y + 1, points)
    if y - 1 >= 0 and x1[x][y - 1] > 0 and not vis[x][y - 1]:
        n += dfs(x1, x, y - 1, points)
    return n


def loss_midu(x1):
    x1 = torch.tanh(x1)
    global vis
    vis = np.zeros((cam_edge, cam_edge))
    loss = []
    for i in range(cam_edge):
        for j in range(cam_edge):
            if x1[i][j] > 0 and not vis[i][j]:
                point = []
                n = dfs(x1, i, j, point)
                loss.append(reduce(lambda x, y: x + y, point) / (cam_edge * cam_edge + 1 - n))
    if len(loss) == 0:
        return torch.tensor(0.0, device=x1.device)  # 返回一个和输入在同一设备上的张量
    return reduce(lambda x, y: x + y, loss) / len(loss)


# --- MODIFICATION 1: 移除 Variable ---
# Textures
texture_param_np = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32') * -0.9
# 直接创建张量并设置 requires_grad=True
texture_param = torch.from_numpy(texture_param_np).cuda(device=0).requires_grad_(True)

texture_origin = torch.from_numpy(textures[None, :, :, :, :, :]).cuda(device=0)
texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
with open(args.faces, 'r') as f:
    face_ids = f.readlines()
    for face_id in face_ids:
        if face_id.strip():  # 更稳健的判断
            texture_mask[int(face_id.strip()) - 1, :, :, :, :] = 1
texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)


def cal_texture(CONTENT=False):
    if CONTENT:
        textures = 0.5 * (torch.nn.Tanh()(texture_content) + 1)
    else:
        textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)
    return texture_origin * (1 - texture_mask) + texture_mask * textures


# 使用 MyDataset 加载训练数据（图像、位姿信息），并通过 DataLoader 批量读取数据。
# MyDataset 会根据相机和车辆位姿渲染 3D 模型的纹理图像，并与原始图像融合（结合掩码）。
def run_cam(data_dir, epoch, train=True, batch_size=BATCH_SIZE):
    print(data_dir)
    dataset = MyDataset(data_dir, 640, texture_size, faces, vertices, distence=None, mask_dir=mask_dir, ret_mask=True)
    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        # num_workers=2,
    )

    optim = torch.optim.Adam([texture_param], lr=LR)
    Cam = CAM()

    # 修改1：增加模型冻结，防止意外修改预训练模型参数
    for param in Cam.model.parameters():
        param.requires_grad = False

    textures = cal_texture()
    dataset.set_textures(textures)
    print(len(dataset))

    for _ in range(epoch):
        print('Epoch: ', _, '/', epoch)
        tqdm_loader = tqdm.tqdm(loader)
        for i, (index, total_img, texture_img, masks) in enumerate(tqdm_loader):
            index = int(index[0])

            # 保存图片的代码只在第一次迭代时执行，避免冗余
            if i == 0 and _ == 0:
                total_img_np = total_img.detach().cpu().numpy()[0]  # 使用 detach()
                total_img_np = Image.fromarray(np.transpose(total_img_np, (1, 2, 0)).astype('uint8'))
                total_img_np.save(os.path.join(log_dir, 'test_total.jpg'))

                Image.fromarray(
                    (255 * texture_img.detach().cpu().numpy()[0].transpose((1, 2, 0))).astype('uint8')).save(
                    os.path.join(log_dir, 'texture2.png'))
                Image.fromarray((255 * masks.detach().cpu().numpy()[0]).astype('uint8')).save(
                    os.path.join(log_dir, 'mask.png'))

            #######
            # CAM #
            #######
            # 修改2：增加数据类型检查，确保输入正确
            if total_img.dtype != torch.float32:
                total_img = total_img.float()
            mask, pred = Cam(total_img, index, log_dir)  # CAM 内部可能需要修改

            ###########
            #   LOSS  #
            ###########
            # 修改3：增加损失项权重调整，便于调参
            loss_midu_weight = 1.0
            loss_content_weight = lamb
            loss_smooth_weight = 1.0
            loss = (loss_midu_weight * loss_midu(mask) +
                    loss_content_weight * loss_content_diff(texture_param) +
                    loss_smooth_weight * loss_smooth(texture_img, masks))

            with open(os.path.join(log_dir, 'loss.txt'), 'a') as f:
                # --- MODIFICATION 2: 使用 .detach() ---
                loss_np = loss.detach().cpu().numpy()
                pred_np = pred.detach().cpu().numpy() if isinstance(pred, torch.Tensor) else pred
                tqdm_loader.set_description('Loss %.8f, Prob %.8f' % (loss_np, pred_np))
                f.write('Loss %.8f, Prob %.8f\n' % (loss_np, pred_np))

            ############
            # backward #
            ############
            if train and loss.item() != 0:  # 使用 .item() 获取标量
                optim.zero_grad()
                # --- MODIFICATION 3: 检查是否需要 retain_graph ---
                # 如果 CAM 的计算图与 loss 的计算图是分离的，可能不需要 retain_graph=True
                # 如果报错 "one of the variables needed for gradient computation has been modified by an inplace operation"
                # 则可能需要加回来，但最好是找到 inplace 操作并修改

                # 修改4：增加梯度裁剪，防止梯度爆炸
                torch.nn.utils.clip_grad_norm_([texture_param], max_norm=1.0)
                loss.backward()
                optim.step()

            textures = cal_texture()
            dataset.set_textures(textures)


if __name__ == '__main__':
    logs = {
        'epoch': EPOCH,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'model': 'resnet50',
        'loss_func': 'loss_midu+loss_content+loss_smooth',
        'lamb': lamb,
        'D1': D1,
        'D2': D2,
        'T': T,
    }

    make_log_dir(logs)
    train_img_dir = os.path.join(args.datapath, 'new_dataset/images/train/')
    # --- MODIFICATION 4: 同样移除这里的 Variable ---
    # 注意：这里的 texture_param 会覆盖全局的 texture_param，如果这是本意的话
    texture_param = torch.from_numpy(np.load(args.content)).cuda(device=0).requires_grad_(True)

    run_cam(train_img_dir, EPOCH)

    # 修改5：保存前增加类型检查
    if isinstance(texture_param, torch.Tensor):
        np.save(os.path.join(log_dir, 'texture.npy'), texture_param.detach().cpu().numpy())
    else:
        np.save(os.path.join(log_dir, 'texture.npy'), texture_param)