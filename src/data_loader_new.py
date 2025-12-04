import torch
from torch.utils.data import Dataset, DataLoader
import os
import sys
import cv2
import numpy as np
import warnings
import trimesh  # 用于加载OBJ文件

warnings.filterwarnings("ignore")

sys.path.append("../renderer/")

import nmr_test_new as nmr
import neural_renderer


class MyDataset(Dataset):
    def __init__(self, data_dir, img_size, texture_size, faces, vertices, distence=None, mask_dir='', ret_mask=False):
        self.data_dir = data_dir
        self.files = []
        # 过滤仅保留npz文件（避免非数据文件干扰）
        files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        for file in files:
            if distence is None:
                self.files.append(file)
            else:
                try:
                    data = np.load(os.path.join(self.data_dir, file))
                    veh_trans = data['veh_trans']
                    cam_trans = data['cam_trans']

                    # 计算相机和车辆的位置关系
                    cam_trans[0][0] += veh_trans[0][0]
                    cam_trans[0][1] += veh_trans[0][1]
                    cam_trans[0][2] += veh_trans[0][2]
                    veh_trans[0][2] += 0.2

                    # 计算距离平方和
                    dis = (cam_trans - veh_trans)[0, :]
                    dis = np.sum(dis ** 2)
                    if dis <= distence:
                        self.files.append(file)
                except Exception as e:
                    print(f"处理文件 {file} 时出错: {e}")
        print(f"加载的样本数量: {len(self.files)}")

        self.img_size = img_size
        # 初始化纹理（根据面数生成）
        textures = np.ones((1, faces.shape[0], texture_size, texture_size, texture_size, 3), 'float32')
        self.textures = torch.from_numpy(textures).cuda(device=0)
        self.faces_var = torch.from_numpy(faces[None, :, :]).cuda(device=0)
        self.vertices_var = torch.from_numpy(vertices[None, :, :]).cuda(device=0)
        self.mask_renderer = nmr.NeuralRenderer(img_size=self.img_size).cuda()
        self.mask_dir = mask_dir
        self.ret_mask = ret_mask
        # 验证掩码目录是否存在
        if ret_mask and not os.path.exists(mask_dir):
            warnings.warn(f"掩码目录 {mask_dir} 不存在，可能导致读取掩码失败")

    def set_textures(self, textures):
        # 修改1：增加纹理设置时的类型检查
        if not isinstance(textures, torch.Tensor):
            textures = torch.from_numpy(textures).cuda(device=0)
        elif textures.device != torch.device('cuda:0'):
            textures = textures.cuda(device=0)
        self.textures = textures

    def __getitem__(self, index):
        file = os.path.join(self.data_dir, self.files[index])
        try:
            data = np.load(file)
            img = data['img']
            veh_trans = data['veh_trans']
            cam_trans = data['cam_trans']

            # 调整相机和车辆位置
            cam_trans[0][0] += veh_trans[0][0]
            cam_trans[0][1] += veh_trans[0][1]
            cam_trans[0][2] += veh_trans[0][2]
            veh_trans[0][2] += 0.2

            # 获取渲染参数
            eye, camera_direction, camera_up = nmr.get_params(cam_trans, veh_trans)
            self.mask_renderer.renderer.renderer.eye = eye
            self.mask_renderer.renderer.renderer.camera_direction = camera_direction
            self.mask_renderer.renderer.renderer.camera_up = camera_up

            # 渲染图像
            imgs_pred = self.mask_renderer.forward(self.vertices_var, self.faces_var, self.textures)

            # 处理原始图像
            img = img[:, :, ::-1]  # BGR转RGB
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = np.transpose(img, (2, 0, 1))  # 转为(C, H, W)
            img = np.expand_dims(img, axis=0)  # 增加批次维度
            img = torch.from_numpy(img).cuda(device=0).float()  # 转为float类型

            # 归一化渲染结果
            # 修改2：改进归一化方式，增加数值稳定性
            max_val = torch.max(imgs_pred)
            if max_val > 0:
                imgs_pred = imgs_pred / (max_val + 1e-8)
            else:
                imgs_pred = torch.zeros_like(imgs_pred)

            # 处理掩码
            mask = None
            if self.ret_mask and self.mask_dir:
                mask_file = os.path.join(self.mask_dir, self.files[index].replace('.npz', '.png'))
                if os.path.exists(mask_file):
                    mask = cv2.imread(mask_file)
                    mask = cv2.resize(mask, (self.img_size, self.img_size))
                    # 合并三通道掩码（取逻辑或）
                    mask = np.logical_or.reduce(mask, axis=2).astype('float32')
                    mask = torch.from_numpy(mask).cuda(device=0)
                else:
                    warnings.warn(f"掩码文件 {mask_file} 不存在，使用全0掩码")
                    mask = torch.zeros((self.img_size, self.img_size), dtype=torch.float32).cuda(device=0)

            # 融合图像
            if mask is not None:
                # 修改3：确保掩码维度正确
                if len(mask.shape) == 2:
                    mask = mask.unsqueeze(0).unsqueeze(0)  # 增加通道和批次维度
                total_img = img * (1 - mask) + 255 * imgs_pred * mask
            else:
                total_img = img  # 无掩码时直接使用原始图像

            return index, total_img.squeeze(0), imgs_pred.squeeze(0), mask

        except Exception as e:
            print(f"处理样本 {self.files[index]} 时出错: {e}")
            # 返回占位数据避免程序中断
            return index, torch.zeros(3, self.img_size, self.img_size).cuda(), torch.zeros(3, self.img_size,
                                                                                           self.img_size).cuda(), torch.zeros(
                self.img_size, self.img_size).cuda()

    def __len__(self):
        return len(self.files)


if __name__ == '__main__':
    # 加载OBJ模型（处理Scene对象的情况）
    obj_file = 'audi_et_te.obj'  # 确保路径正确，建议使用绝对路径如'/path/to/audi_et_te.obj'
    if not os.path.exists(obj_file):
        raise FileNotFoundError(f"OBJ文件 {obj_file} 不存在，请检查路径")

    # 加载模型并提取网格数据
    mesh_or_scene = trimesh.load(obj_file)
    if isinstance(mesh_or_scene, trimesh.Scene):
        # 从场景中提取第一个网格（根据实际模型调整）
        meshes = list(mesh_or_scene.geometry.values())
        if not meshes:
            raise ValueError("OBJ文件中未包含任何网格数据")
        mesh = meshes[0]
        print(f"从场景中提取网格: {mesh}")
    else:
        mesh = mesh_or_scene

    # 提取顶点和面数据
    vertices = mesh.vertices.astype(np.float32)
    faces = mesh.faces.astype(np.int32)
    print(f"顶点数量: {vertices.shape}, 面数量: {faces.shape}")

    # 数据集路径（请根据实际情况修改）
    train_img_dir = "data/phy_attack/images/train/"
    train_mask_dir = "data/masks/train/"

    # 验证数据目录是否存在
    if not os.path.exists(train_img_dir):
        raise NotADirectoryError(f"训练数据目录 {train_img_dir} 不存在")
    if not os.path.exists(train_mask_dir):
        raise NotADirectoryError(f"掩码目录 {train_mask_dir} 不存在")

    # 初始化数据集
    dataset = MyDataset(
        data_dir=train_img_dir,
        img_size=640,
        texture_size=4,
        faces=faces,
        vertices=vertices,
        mask_dir=train_mask_dir,
        ret_mask=True
    )

    # 数据加载器
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=True,
        drop_last=True  # 丢弃不完整的批次
    )

    # 测试数据加载
    try:
        for index, total_img, imgs_pred, mask in loader:
            print(f"\n成功加载样本: index={index.item()}")
            print(f"融合图像尺寸: {total_img.shape}")
            print(f"渲染图像尺寸: {imgs_pred.shape}")
            print(f"掩码尺寸: {mask.shape}")
            print(f"融合图像值范围: [{total_img.min():.2f}, {total_img.max():.2f}]")
            break  # 只测试第一个样本
    except Exception as e:
        print(f"数据加载测试失败: {e}")