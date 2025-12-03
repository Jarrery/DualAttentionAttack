import os
import sys
from PIL import Image
import numpy as np
import tqdm
import torch
import cv2
import warnings
from sklearn.metrics import average_precision_score
import json  # 用于加载目标框标注

warnings.filterwarnings("ignore")

import torch.nn.functional as F
import nmr_test as nmr
import neural_renderer
from torchvision.transforms import Resize, ToPILImage
from data_loader_new import MyDataset
from torch.utils.data import DataLoader
from torchvision import models, transforms

from argparse import ArgumentParser

# --------------------------
# 配置参数
# --------------------------
parser = ArgumentParser()
parser.add_argument("--texture", type=str, default='./logs/epoch-640&640/texture.npy', help="生成的纹理文件路径")
parser.add_argument("--obj", type=str, default='audi_et_te.obj', help="车辆模型OBJ文件")
parser.add_argument("--datapath", type=str, default='./data', help="数据根目录")
parser.add_argument("--annotations", type=str, default='./data/annotations.json', help="目标框标注文件（用于AP计算）")
args = parser.parse_args()

# 初始化路径
mask_dir = os.path.join(args.datapath, 'masks/')
log_dir = './evaluation_results'
os.makedirs(log_dir, exist_ok=True)

# 模型和纹理参数
texture_size = 6
vertices, faces, textures = neural_renderer.load_obj(
    filename_obj=args.obj,
    texture_size=texture_size,
    load_texture=True
)
label_list = [468, 511, 609, 817, 581, 751, 627]  # 目标类别列表


# --------------------------
# 1. 纹理加载与贴装
# --------------------------
def load_trained_texture(texture_path):
    """加载训练好的纹理并应用到车辆模型"""
    # 加载纹理参数
    texture_param_np = np.load(texture_path)
    texture_param = torch.from_numpy(texture_param_np).cuda(device=0).requires_grad_(False)  # 测试阶段不需要梯度

    # 纹理掩码（控制哪些面需要贴装纹理）
    texture_mask = np.zeros((faces.shape[0], texture_size, texture_size, texture_size, 3), 'int8')
    with open('all_faces.txt', 'r') as f:
        for face_id in f.readlines():
            if face_id.strip():
                texture_mask[int(face_id.strip()) - 1, :, :, :, :] = 1
    texture_mask = torch.from_numpy(texture_mask).cuda(device=0).unsqueeze(0)

    # 原始纹理（未修改的部分）
    texture_origin = torch.from_numpy(textures[None, :, :, :, :, :]).cuda(device=0)

    # 合并纹理（仅修改掩码指定的面）
    textures = 0.5 * (torch.nn.Tanh()(texture_param) + 1)  # 反归一化到[0,1]
    final_texture = texture_origin * (1 - texture_mask) + texture_mask * textures
    return final_texture


# --------------------------
# 2. 指标计算函数
# --------------------------
class MetricsCalculator:
    def __init__(self, annotations_path):
        self.annotations = self.load_annotations(annotations_path)
        self.all_preds = []  # 存储所有预测置信度
        self.all_labels = []  # 存储所有真实标签（1=正样本，0=负样本）

    def load_annotations(self, path):
        """加载目标框标注（格式：{image_id: {'boxes': [[x1,y1,x2,y2]], 'category_id': int}}）"""
        with open(path, 'r') as f:
            return json.load(f)

    def calculate_iou(self, box1, box2):
        """计算两个边界框的IoU"""
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2
        xi1, yi1 = max(x1, x1g), max(y1, y1g)
        xi2, yi2 = min(x2, x2g), min(y2, y2g)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2g - x1g) * (y2g - y1g)
        union_area = box1_area + box2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def update_ap(self, image_id, pred_box, pred_score, pred_category):
        """更新AP计算所需的预测和标签"""
        if image_id not in self.annotations:
            return
        gt = self.annotations[image_id]
        # 真实标签：1=目标类别，0=其他
        gt_label = 1 if gt['category_id'] in label_list else 0
        # 预测置信度（归一化到[0,1]）
        pred_score = float(pred_score)
        # 判断是否为正样本（IoU>=0.5且类别正确）
        if gt_label == 1:
            iou = self.calculate_iou(pred_box, gt['boxes'][0])
            pred_label = 1 if (iou >= 0.5 and pred_category in label_list) else 0
        else:
            pred_label = 0  # 负样本不参与IoU计算

        self.all_preds.append(pred_score)
        self.all_labels.append(pred_label)

    def get_ap_at_05(self):
        """计算AP@0.5"""
        if not self.all_preds or not self.all_labels:
            return 0.0
        return average_precision_score(self.all_labels, self.all_preds)

    def calculate_asr(self, success_count, total_count):
        """计算ASR（攻击成功率）"""
        return success_count / total_count if total_count > 0 else 0.0


# --------------------------
# 3. 模型测试与指标评估
# --------------------------
def test_model(data_dir, texture, model_name, model):
    """测试模型并计算AP@0.5和ASR"""
    # 初始化数据集（使用贴装后的纹理）
    dataset = MyDataset(
        data_dir=data_dir,
        img_size=224,  # 可根据需要改为2048
        texture_size=texture_size,
        faces=faces,
        vertices=vertices,
        distence=50,
        mask_dir=mask_dir,
        ret_mask=True
    )
    dataset.set_textures(texture)  # 应用训练好的纹理
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 初始化指标计算器
    metrics = MetricsCalculator(args.annotations)
    success_count = 0  # 攻击成功计数（错误分类）

    # 测试过程
    tqdm_loader = tqdm.tqdm(loader, desc=f"Testing {model_name}")
    for i, (index, total_img, texture_img, masks) in enumerate(tqdm_loader):
        index = int(index[0])
        image_id = dataset.files[index].replace('.npz', '')  # 图像ID（与标注对应）

        # 保存渲染结果（贴装纹理后的车辆）
        img_np = total_img.detach().cpu().numpy()[0]
        img_pil = Image.fromarray(np.transpose(img_np, (1, 2, 0)).astype('uint8'))
        img_pil.save(os.path.join(log_dir, f"{model_name}_texture_{image_id}.jpg"))

        # 图像预处理
        preprocessed_img = preprocess_image(total_img / 255)

        # 模型预测
        with torch.no_grad():
            output = model(preprocessed_img)
            prob = F.softmax(output, dim=1)
            pred_score, pred_category = prob.max(dim=1)
            pred_category = int(pred_category.cpu().numpy())
            pred_score = float(pred_score.cpu().numpy())

        # 计算ASR：预测类别不在目标列表则视为攻击成功
        if pred_category not in label_list:
            success_count += 1

        # 计算AP@0.5：需要目标检测框（这里简化为全图框，实际应替换为模型输出的检测框）
        pred_box = [0, 0, 224, 224]  # 假设目标框覆盖全图（需根据实际检测结果修改）
        metrics.update_ap(image_id, pred_box, pred_score, pred_category)

    # 输出指标
    asr = metrics.calculate_asr(success_count, len(loader))
    ap = metrics.get_ap_at_05()
    print(f"{model_name} - ASR: {asr:.4f}, AP@0.5: {ap:.4f}")

    # 保存结果
    with open(os.path.join(log_dir, 'metrics.txt'), 'a') as f:
        f.write(f"{model_name}, ASR: {asr:.4f}, AP@0.5: {ap:.4f}\n")


def preprocess_image(img):
    """图像预处理（与训练时保持一致）"""
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for i in range(3):
        img[:, i, :, :] = (img[:, i, :, :] - means[i]) / stds[i]
    return img


# --------------------------
# 主函数
# --------------------------
if __name__ == '__main__':
    # 1. 加载训练好的纹理并贴到车辆模型
    trained_texture = load_trained_texture(args.texture)
    print(f"成功加载纹理并贴装到模型: {args.obj}")

    # 2. 加载测试数据目录
    test_dir = os.path.join(args.datapath, 'phy_attack/images/train')

    # 3. 定义测试模型
    models_dict = {
        'resnet152': models.resnet152(pretrained=True).eval().cuda(),
        'densenet201': models.densenet201(pretrained=True).eval().cuda(),
        'vgg19': models.vgg19(pretrained=True).eval().cuda(),
        'inception_v3': models.inception_v3(pretrained=True).eval().cuda()
    }

    # 4. 逐个测试模型并计算指标
    for name, model in models_dict.items():
        test_model(test_dir, trained_texture, name, model)