import os
import sys
import numpy as np
import tqdm
import torch
import cv2
import warnings
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Resize
import torch.nn.functional as F
import argparse
from functools import reduce

# 引入 YOLOv7 依赖（需将 YOLOv7 仓库的 models/ 和 utils/ 放入项目路径）
sys.path.append("./src/yolov7")  # 替换为你的 YOLOv7 实际目录
from yolov7.models.experimental import attempt_load
from yolov7.utils.general import non_max_suppression, scale_coords, bbox_iou
from yolov7.utils.torch_utils import select_device
import neural_renderer
import nmr_test as nmr  # 原项目的神经渲染器

warnings.filterwarnings("ignore")
torch.manual_seed(2333)
torch.cuda.manual_seed(2333)
np.random.seed(2333)

# ======================== 1. 命令行参数定义（完整） ========================
parser = argparse.ArgumentParser()
# 训练基础参数
parser.add_argument("--epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--batchsize", type=int, default=8)
# 损失函数权重
parser.add_argument("--lamb", type=float, default=5e-3)  # 模型注意力损失权重
parser.add_argument("--mu", type=float, default=1e-4)  # 人类注意力损失权重
parser.add_argument("--d1", type=float, default=50)  # 边缘区域保护系数
parser.add_argument("--d2", type=float, default=10)  # 非边缘区域保护系数
parser.add_argument("--t", type=float, default=0.1)  # 平滑损失系数
# 3D模型与数据路径
parser.add_argument("--obj", type=str, default='audi_et_te.obj')
parser.add_argument("--faces", type=str, default='./all_faces.txt')
parser.add_argument("--datapath", type=str, required=True)  # 自定义数据集根目录（必填）
parser.add_argument("--content", type=str, required=True)  # 初始纹理路径（必填）
parser.add_argument("--canny", type=str, required=True)  # 边缘 mask 路径（必填）
# 纹理与检测参数
parser.add_argument("--texture_size", type=int, default=6)  # 原项目固定纹理尺寸
parser.add_argument("--conf_thres", type=float, default=0.5)  # 检测置信度阈值
parser.add_argument("--img_size", type=int, default=640)  # 输入图像尺寸
args = parser.parse_args()

# ======================== 2. 设备与模型初始化 ========================
# 初始化设备
device = select_device('0' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 初始化 YOLOv7 模型（黑盒攻击模式，冻结权重）
yolov7_model = attempt_load('yolov7.pt', map_location=device)  # 替换为你的YOLOv7权重路径
yolov7_model.eval()
for param in yolov7_model.parameters():
    param.requires_grad = False

# 3D 模型与纹理初始化
obj_file = args.obj
vertices, faces, textures = neural_renderer.load_obj(
    filename_obj=obj_file,
    texture_size=args.texture_size,
    load_texture=True
)
# 转换为CUDA张量
vertices = torch.from_numpy(vertices).float().cuda(device)
faces = torch.from_numpy(faces).int().cuda(device)
textures = torch.from_numpy(textures).float().cuda(device)


# ======================== 3. 日志与纹理参数初始化 ========================
# 日志目录设置
def make_log_dir():
    dir_name = (f"logs/epoch-{args.epoch}_lr-{args.lr}_bs-{args.batchsize}_"
                f"lamb-{args.lamb}_mu-{args.mu}_d1-{args.d1}_d2-{args.d2}_t-{args.t}")
    os.makedirs(dir_name, exist_ok=True)
    return dir_name


log_dir = make_log_dir()
print(f"日志保存目录: {log_dir}")

# 加载初始纹理与边缘mask
texture_content = torch.from_numpy(np.load(args.content)).float().cuda(device)
texture_canny = torch.from_numpy(np.load(args.canny)).float().cuda(device)
texture_canny = (texture_canny >= 1).int()  # 边缘 mask 二值化

# 可训练面掩码（与原项目一致）
texture_mask = np.zeros(
    (faces.shape[0], args.texture_size, args.texture_size, args.texture_size, 3),
    dtype=np.float32
)
with open(args.faces, 'r') as f:
    for line in f:
        if line.strip():
            face_id = int(line.strip()) - 1  # 面ID从1开始，转换为0索引
            texture_mask[face_id] = 1.0
texture_mask = torch.from_numpy(texture_mask).float().cuda(device).unsqueeze(0)
texture_origin = textures[None, ...]  # 原始纹理备份


# ======================== 4. 自定义数据集（适配原项目逻辑） ========================
class CustomDataset(Dataset):
    def __init__(self, data_dir, img_size=640, texture_size=6, faces=None, vertices=None, mask_dir=''):
        # 原项目数据路径适配
        self.base_dir = data_dir
        self.img_dir = os.path.join(data_dir, 'phy_attack/images/train/')
        self.label_dir = os.path.join(data_dir, 'phy_attack/labels/train/')
        self.mask_dir = mask_dir if mask_dir else os.path.join(data_dir, 'phy_attack/masks/')

        # 筛选npz数据文件（原项目数据格式）
        self.files = [f for f in os.listdir(self.img_dir) if f.endswith('.npz')]
        if len(self.files) == 0:
            raise ValueError(f"未在 {self.img_dir} 找到npz数据文件")

        self.img_size = img_size
        self.texture_size = texture_size
        self.faces = faces
        self.vertices = vertices

        # 初始化神经渲染器
        self.renderer = nmr.NeuralRenderer(img_size=img_size).cuda(device)
        self.textures = None  # 动态更新的对抗纹理

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 加载npz文件（包含图像、位姿等信息）
        file_name = self.files[idx]
        npz_path = os.path.join(self.img_dir, file_name)
        data = np.load(npz_path)

        # 提取核心数据
        img = data['img']  # 原始图像 (H,W,3)
        veh_trans = data['veh_trans']  # 车辆位姿
        cam_trans = data['cam_trans']  # 相机位姿

        # 计算相机参数（复用原项目逻辑）
        eye, camera_direction, camera_up = nmr.get_params(cam_trans, veh_trans)
        self.renderer.renderer.renderer.eye = eye
        self.renderer.renderer.renderer.camera_direction = camera_direction
        self.renderer.renderer.renderer.camera_up = camera_up

        # 渲染对抗纹理图像
        rendered_img = torch.zeros(3, self.img_size, self.img_size).float()
        if self.textures is not None:
            rendered_img = self.renderer.forward(
                self.vertices[None, :, :],
                self.faces[None, :, :],
                self.textures
            ).squeeze(0)

        # 加载掩码并预处理
        mask_file = os.path.join(self.mask_dir, file_name.replace('.npz', '.png'))
        mask = cv2.imread(mask_file, 0) / 255.0  # 灰度掩码 [0,1]
        mask = cv2.resize(mask, (self.img_size, self.img_size))
        mask = torch.from_numpy(mask).float()

        # 融合原始图像与渲染图像（原项目核心逻辑）
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # (3,H,W) [0,1]
        total_img = img * (1 - mask) + rendered_img * mask

        # 加载YOLO格式标签
        label_file = os.path.join(self.label_dir, file_name.replace('.npz', '.txt'))
        if os.path.exists(label_file):
            labels = np.loadtxt(label_file).reshape(-1, 5)  # (n,5) [cls, xc, yc, w, h]
        else:
            labels = np.zeros((0, 5))

        return {
            'img': total_img,  # 融合后的图像 (3,H,W)
            'labels': torch.from_numpy(labels).float(),  # YOLO标签
            'mask': mask,  # 车辆掩码 (H,W)
            'file_name': file_name,  # 文件名（用于日志）
            'index': idx  # 索引
        }

    def set_textures(self, textures):
        """更新对抗纹理"""
        self.textures = textures


# ======================== 5. 损失函数定义（适配YOLOv7） ========================
def loss_human_attention(tex):
    """人类注意力规避损失（边缘约束）"""
    # 只计算可训练区域的损失
    tex = tex * texture_mask
    edge_loss = args.d1 * torch.sum(texture_canny * torch.pow(tex - texture_content, 2))
    non_edge_loss = args.d2 * torch.sum((1 - texture_canny) * torch.pow(tex - texture_content, 2))
    return args.mu * (edge_loss + non_edge_loss)


def loss_smooth(img, mask=None):
    """纹理平滑损失（带掩码）"""
    if mask is None:
        mask = torch.ones_like(img[:, :-1, :-1])
    else:
        mask = mask[:-1, :-1].unsqueeze(0).repeat(3, 1, 1)  # 扩展到3通道

    # 计算水平/垂直方向的平滑损失
    s1 = torch.pow(img[:, :, 1:, :-1] - img[:, :, :-1, :-1], 2)
    s2 = torch.pow(img[:, :, :-1, 1:] - img[:, :, :-1, :-1], 2)
    return args.t * torch.sum(mask * (s1 + s2))


def loss_model_attention(pred_boxes, img_size=640):
    """YOLOv7模型注意力损失：最小化车辆目标置信度"""
    loss = 0.0
    valid_count = 0

    for pred in pred_boxes:
        if len(pred) == 0:
            continue

        # 筛选车辆类别（假设类别0为车辆，根据你的数据集调整）
        vehicle_preds = pred[pred[:, 5] == 0]
        if len(vehicle_preds) > 0:
            # 目标：降低车辆检测置信度
            loss += torch.mean(vehicle_preds[:, 4])
            valid_count += 1

    return args.lamb * (loss / max(valid_count, 1))


def total_loss(rendered_imgs, true_labels, texture_param, masks):
    """总损失：模型注意力损失 + 人类注意力损失 + 平滑损失"""
    # 1. YOLOv7推理（禁用梯度）
    with torch.no_grad():
        pred = yolov7_model(rendered_imgs)[0]
        pred_boxes = non_max_suppression(
            pred,
            conf_thres=args.conf_thres,
            iou_thres=0.45,
            classes=None,  # 检测所有类别
            agnostic=True
        )

    # 2. 计算各部分损失
    loss_attn = loss_model_attention(pred_boxes, args.img_size)  # 模型注意力损失
    loss_human = loss_human_attention(texture_param)  # 人类注意力损失
    loss_smooth_ = loss_smooth(rendered_imgs, masks)  # 平滑损失

    # 总损失
    total = loss_attn + loss_human + loss_smooth_

    # 记录各损失分量（用于日志）
    return total, {
        'model_attention': loss_attn.item(),
        'human_attention': loss_human.item(),
        'smooth': loss_smooth_.item()
    }


# ======================== 6. 训练主函数 ========================
def train():
    # 初始化数据集
    dataset = CustomDataset(
        data_dir=args.datapath,
        img_size=args.img_size,
        texture_size=args.texture_size,
        faces=faces,
        vertices=vertices,
        mask_dir=os.path.join(args.datapath, 'phy_attack/masks/')
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,  # 原项目不打乱数据
        num_workers=4,  # 根据CPU核心数调整
        pin_memory=True
    )
    print(f"数据集加载完成，共 {len(dataset)} 个样本")

    # 初始化可训练纹理参数
    texture_param = torch.from_numpy(np.load(args.content)).float().cuda(device)
    texture_param.requires_grad = True  # 开启梯度

    # 优化器（与原项目一致）
    optimizer = torch.optim.Adam([texture_param], lr=args.lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 训练循环
    for epoch in range(args.epoch):
        dataset.set_textures(None)  # 重置纹理
        total_loss_epoch = 0.0
        loss_components = {'model_attention': 0.0, 'human_attention': 0.0, 'smooth': 0.0}

        # 计算当前对抗纹理（原项目cal_texture逻辑）
        current_texture = 0.5 * (torch.tanh(texture_param) + 1)  # 归一化到[0,1]
        current_texture = texture_origin * (1 - texture_mask) + texture_mask * current_texture
        dataset.set_textures(current_texture)

        # 批次训练
        pbar = tqdm.tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epoch}")
        for batch in pbar:
            # 加载批次数据
            rendered_imgs = batch['img'].cuda(device)
            true_labels = batch['labels'].cuda(device)
            masks = batch['mask'].cuda(device)

            # 计算损失
            loss, loss_dict = total_loss(rendered_imgs, true_labels, texture_param, masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([texture_param], max_norm=1.0)  # 梯度裁剪
            optimizer.step()

            # 累计损失
            total_loss_epoch += loss.item()
            for k in loss_components.keys():
                loss_components[k] += loss_dict[k]

            # 更新进度条
            pbar.set_postfix({
                'total_loss': f"{loss.item():.4f}",
                'model_attn': f"{loss_dict['model_attention']:.4f}",
                'human_attn': f"{loss_dict['human_attention']:.4f}",
                'smooth': f"{loss_dict['smooth']:.4f}"
            })

        # 学习率调度
        scheduler.step()

        # 计算epoch平均损失
        avg_loss = total_loss_epoch / len(dataloader)
        avg_loss_components = {k: v / len(dataloader) for k, v in loss_components.items()}

        # 保存日志
        with open(os.path.join(log_dir, "loss_log.txt"), "a") as f:
            f.write(f"Epoch {epoch + 1}, AvgLoss: {avg_loss:.4f}, "
                    f"ModelAttn: {avg_loss_components['model_attention']:.4f}, "
                    f"HumanAttn: {avg_loss_components['human_attention']:.4f}, "
                    f"Smooth: {avg_loss_components['smooth']:.4f}\n")

        # 保存纹理参数
        texture_save_path = os.path.join(log_dir, f"texture_epoch_{epoch + 1}.npy")
        np.save(texture_save_path, texture_param.detach().cpu().numpy())
        print(f"Epoch {epoch + 1} 完成，平均损失: {avg_loss:.4f}，纹理保存至: {texture_save_path}")

    # 保存最终纹理
    final_texture_path = os.path.join(log_dir, "texture_final.npy")
    np.save(final_texture_path, texture_param.detach().cpu().numpy())
    print(f"训练完成！最终纹理保存至: {final_texture_path}")


# ======================== 7. 主函数入口 ========================
if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"训练过程出错: {e}")
        raise e