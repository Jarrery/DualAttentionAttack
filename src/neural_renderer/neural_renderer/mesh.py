import torch
import torch.nn as nn
# 假设使用pytorch3d加载obj文件
from pytorch3d.io import load_obj


class Mesh(nn.Module):
    def __init__(self, filename_obj, texture_size=4, normalization=True):
        super(Mesh, self).__init__()

        # 加载.obj文件（pytorch3d的load_obj返回顶点、面和辅助信息）
        vertices, faces, _ = load_obj(filename_obj, load_textures=False)
        self.faces = faces.verts_idx  # 面的顶点索引
        self.num_vertices = vertices.shape[0]
        self.num_faces = self.faces.shape[0]

        # 顶点参数（可学习）
        self.vertices = nn.Parameter(vertices.float())

        # 纹理初始化（正态分布）
        self.texture_size = texture_size
        texture_shape = (self.num_faces, texture_size, texture_size, texture_size, 3)
        self.textures = nn.Parameter(torch.randn(*texture_shape) * 0.01)  # 类似Normal初始化

        # 归一化处理（保持原逻辑）
        if normalization:
            vertices = vertices - vertices.min(dim=0)[0]
            vertices = vertices / vertices.abs().max()
            vertices = vertices * 2
            vertices = vertices - vertices.max(dim=0)[0] / 2

    def to(self, device):
        super().to(device)
        self.faces = self.faces.to(device)  # 面索引迁移到设备
        return self

    def get_batch(self, batch_size):
        # 广播到批次维度
        vertices = self.vertices.unsqueeze(0).repeat(batch_size, 1, 1)
        faces = self.faces.unsqueeze(0).repeat(batch_size, 1, 1)
        textures = torch.sigmoid(self.textures.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1, 1))
        return vertices, faces, textures

    def set_lr(self, lr_vertices, lr_textures):
        # PyTorch中通过优化器参数组设置学习率，此处记录用于后续优化器配置
        self.lr_vertices = lr_vertices
        self.lr_textures = lr_textures