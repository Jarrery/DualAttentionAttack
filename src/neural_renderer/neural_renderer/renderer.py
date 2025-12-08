import math
import torch
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRenderer,
    MeshRasterizer, SoftPhongShader, TexturesVolume,
    DirectionalLights, AmbientLights, BlendParams
)
from pytorch3d.structures import Meshes
from pytorch3d.ops import interpolate_face_attributes


class Renderer:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # 渲染设置
        self.image_size = 256
        self.anti_aliasing = True
        self.background_color = torch.tensor([0.0, 0.0, 0.0], device=device)
        self.fill_back = True

        # 相机参数
        self.perspective = True
        self.viewing_angle = 30.0
        self.eye = torch.tensor(
            [0, 0, -(1.0 / math.tan(math.radians(self.viewing_angle)) + 1)],
            device=device,
            dtype=torch.float32
        )
        self.camera_mode = 'look_at'
        self.camera_direction = torch.tensor([0, 0, 1], device=device, dtype=torch.float32)
        self.camera_up = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)
        self.near = 0.1
        self.far = 100.0

        # 光照参数
        self.light_intensity_ambient = 0.5
        self.light_intensity_directional = 0.5
        self.light_color_ambient = torch.tensor([1.0, 1.0, 1.0], device=device)
        self.light_color_directional = torch.tensor([1.0, 1.0, 1.0], device=device)
        self.light_direction = torch.tensor([0, 1, 0], device=device, dtype=torch.float32)

        # 光栅化参数
        self.rasterizer_eps = 1e-3
        self.device = device

        # 初始化渲染器组件
        self._init_cameras()
        self._init_lights()
        self._init_renderer()

    def _init_cameras(self):
        """初始化透视相机"""
        self.cameras = PerspectiveCameras(
            device=self.device,
            fov=torch.tensor([self.viewing_angle], device=self.device),
            R=torch.eye(3, device=self.device).unsqueeze(0),
            T=self.eye.unsqueeze(0)
        )

    def _init_lights(self):
        """初始化光源（环境光+方向光）"""
        self.ambient_light = AmbientLights(
            device=self.device,
            ambient_color=(self.light_intensity_ambient * self.light_color_ambient).unsqueeze(0)
        )
        self.directional_light = DirectionalLights(
            device=self.device,
            direction=self.light_direction.unsqueeze(0),
            diffuse_color=(self.light_intensity_directional * self.light_color_directional).unsqueeze(0)
        )

    def _init_renderer(self):
        """初始化PyTorch3D渲染器"""
        raster_settings = RasterizationSettings(
            image_size=self.image_size,
            blur_radius=0.05 if self.anti_aliasing else 0.0,
            faces_per_pixel=8 if self.anti_aliasing else 1,
            z_clip_value=self.far,
            perspective_correct=True,
        )

        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=raster_settings
        )

        blend_params = BlendParams(
            background_color=self.background_color.tolist()
        )

        self.shader = SoftPhongShader(
            device=self.device,
            cameras=self.cameras,
            lights=self.ambient_light,  # 后续会合并方向光
            blend_params=blend_params
        )

        self.renderer = MeshRenderer(
            rasterizer=self.rasterizer,
            shader=self.shader
        )

    def _transform_vertices(self, vertices):
        """应用视角变换和透视变换"""
        batch_size = vertices.shape[0]
        eye = self.eye.repeat(batch_size, 1)
        at = torch.zeros_like(eye)
        up = self.camera_up.repeat(batch_size, 1)

        # 计算相机旋转矩阵 (look_at 变换)
        z_axis = torch.nn.functional.normalize(at - eye, dim=1)
        x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis, dim=1), dim=1)
        y_axis = torch.nn.functional.normalize(torch.cross(z_axis, x_axis, dim=1), dim=1)

        # 构建旋转矩阵 [batch_size, 3, 3]
        R = torch.stack([x_axis, y_axis, z_axis], dim=1).transpose(1, 2)
        T = -torch.bmm(R, eye.unsqueeze(2)).squeeze(2)

        # 更新相机参数
        self.cameras.R = R
        self.cameras.T = T

        # 透视变换由PyTorch3D相机自动处理
        return vertices

    def _fill_back_faces(self, faces, textures=None):
        """填充背面（翻转三角形）"""
        if not self.fill_back:
            return faces, textures

        # 翻转三角形顶点顺序
        flipped_faces = faces.flip(dims=[2])
        new_faces = torch.cat([faces, flipped_faces], dim=1)

        if textures is not None:
            # 翻转纹理坐标
            flipped_textures = textures.transpose(2, 4)  # 对应原代码的维度翻转
            new_textures = torch.cat([textures, flipped_textures], dim=1)
            return new_faces, new_textures

        return new_faces, None

    def render_silhouettes(self, vertices, faces):
        """渲染轮廓图"""
        # 处理背面
        faces, _ = self._fill_back_faces(faces)

        # 顶点变换
        vertices = self._transform_vertices(vertices)

        # 创建网格对象（使用默认纹理，只用于轮廓提取）
        meshes = Meshes(verts=vertices, faces=faces)

        # 光栅化获取深度图，转换为轮廓
        fragments = self.rasterizer(meshes)
        silhouettes = (fragments.zbuf > 0).float()  # z>0表示有物体
        return silhouettes

    def render_depth(self, vertices, faces):
        """渲染深度图"""
        # 处理背面
        faces, _ = self._fill_back_faces(faces)

        # 顶点变换
        vertices = self._transform_vertices(vertices)

        # 创建网格对象
        meshes = Meshes(verts=vertices, faces=faces)

        # 光栅化获取深度图
        fragments = self.rasterizer(meshes)
        return fragments.zbuf

    def render(self, vertices, faces, textures):
        """渲染带纹理的彩色图"""
        # 处理背面
        faces, textures = self._fill_back_faces(faces, textures)

        # 顶点变换
        vertices = self._transform_vertices(vertices)

        # 转换纹理格式为PyTorch3D的TexturesVolume
        # 原纹理形状: [batch, num_faces, ts, ts, ts, 3] -> 转换为 [batch, num_faces, 3, ts, ts, ts]
        textures_vol = TexturesVolume(volume=textures.permute(0, 1, 5, 2, 3, 4))

        # 创建网格对象
        meshes = Meshes(
            verts=vertices,
            faces=faces,
            textures=textures_vol
        )

        # 合并环境光和方向光效果
        self.shader.lights = self.ambient_light + self.directional_light

        # 渲染
        images = self.renderer(meshes)
        return images[..., :3]  # 返回RGB通道（去除Alpha通道）