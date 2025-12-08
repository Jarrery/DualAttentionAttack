import chainer
import chainer.functions as cf

import neural_renderer


import torch
import torch.nn.functional as F

def look_at(vertices, eye, at=None, up=None):
    """视角变换：将顶点转换到相机坐标系"""
    assert vertices.ndim == 3  # [batch_size, num_vertices, 3]
    batch_size = vertices.shape[0]
    device = vertices.device

    # 默认参数
    if at is None:
        at = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
    if up is None:
        up = torch.tensor([0, 1, 0], dtype=torch.float32, device=device)

    # 扩展到批次维度
    if eye.ndim == 1:
        eye = eye.unsqueeze(0).repeat(batch_size, 1)
    if at.ndim == 1:
        at = at.unsqueeze(0).repeat(batch_size, 1)
    if up.ndim == 1:
        up = up.unsqueeze(0).repeat(batch_size, 1)

    # 计算相机坐标系轴
    z_axis = F.normalize(at - eye, dim=-1)  # 相机朝向
    x_axis = F.normalize(torch.cross(up, z_axis, dim=-1), dim=-1)  # 右方向
    y_axis = F.normalize(torch.cross(z_axis, x_axis, dim=-1), dim=-1)  # 上方向

    # 旋转矩阵 [batch_size, 3, 3]
    r = torch.stack([x_axis, y_axis, z_axis], dim=1)  # 按列拼接

    # 应用旋转和平移
    vertices = vertices - eye.unsqueeze(1)  # 平移到相机原点
    vertices = torch.matmul(vertices, r.transpose(1, 2))  # 旋转（转置矩阵实现右乘）
    return vertices