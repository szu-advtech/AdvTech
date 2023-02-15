import torch
from torch import nn
import tinycudann as tcnn
import vren
from utils.custom_utils import TruncExp
import numpy as np
from einops import rearrange, repeat, reduce
from utils.custom_utils import VolumeRenderer_REN
from utils.render_utils import NEAR_DISTANCE


class NGPNetwork(nn.Module):

    def __init__(self, scale, dataset_name):
        """
        初始化模型
        :param scale:  场景比例
        """
        super().__init__()
        # 场景比例
        self.scale = scale
        # 数据类型
        self.dataset_name = dataset_name
        # 场景中心
        self.register_buffer('center', torch.zeros(1, 3))
        # xyz轴最小值
        self.register_buffer('xyz_min', -torch.ones(1, 3) * scale)
        # xyz轴最大值
        self.register_buffer('xyz_max', torch.ones(1, 3) * scale)
        # 体素空间一半大小
        self.register_buffer('half_size', (self.xyz_max - self.xyz_min) / 2)
        # 渲染方程
        self.VolumeRenderer_REN = VolumeRenderer_REN()

        # 网格边框大小 [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        # K = 1 适用于 synthetic NeRF，对于大场景，K [1,5]
        self.cascades = max(1 + int(np.ceil(np.log2(2 * scale))), 1)  # max (log2*0.5 向上取整，1)
        # 网格大小
        self.grid_size = 128
        # 网格块 128*128*128 / 8  [262144]
        self.register_buffer('density_bitfield',
                             torch.zeros(self.cascades * self.grid_size ** 3 // 8, dtype=torch.uint8))
        # constants
        # L 特定分辨率体素对应的编码层
        # T 每一层哈希表大小
        # N_min 粗分辨率 16
        # N_max 细分辨率 2048*scale
        # F 每个条目的特征维度数
        L = 16;
        F = 2;
        log2_T = 19;
        N_min = 16
        b = np.exp(np.log(2048 * scale / N_min) / (L - 1))  # e^log(N_max/N_min)/(L-1)
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')
        # 视图方向投影到球谐基的前16个系数(即4阶)上
        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",  # 球面谐波 球函数
                    "degree": 4,  # 阶数
                },  # 位置编码 4阶球函数
            )
        # 密度MLP  输入哈希编码位置y = enc(x;𝜃)  输出16位对数空间密度
        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,  # 输入维度 3 输出维度 16（多分辨率 线性插值 串联）
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },  # 位置编码：多分辨率 体素网格 hash表存储
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }  # 小MLP网络
            )
        if "rffr" == self.dataset_name:
            self.t_rgb = \
                tcnn.Network(
                    n_input_dims=32, n_output_dims=3,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "Sigmoid",
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    }
                )
            self.r_rgb = \
                tcnn.Network(
                    n_input_dims=16, n_output_dims=4,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "Sigmoid",
                        "n_neurons": 64,
                        "n_hidden_layers": 1,
                    }
                )
        else:
            self.rgb_net = \
                tcnn.Network(
                    n_input_dims=32, n_output_dims=3,
                    network_config={
                        "otype": "FullyFusedMLP",
                        "activation": "ReLU",
                        "output_activation": "Sigmoid",
                        "n_neurons": 64,
                        "n_hidden_layers": 2,
                    }
                )

    def density(self, x, return_feat=False):
        """
        获取透明度
        :param x: (N, 3) xyz in [-scale, scale]
        :param return_feat: 是否返回中间特征
        :return:
            sigmas: (N)
        """
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)  # 归一化
        h = self.xyz_encoder(x)  # 获取32位位置编码
        sigmas = TruncExp.apply(h[:, 0])
        if "rffr" == self.dataset_name:
            sigmas = TruncExp.apply(h[:, 0])
            t_beta = TruncExp.apply(h[:, 1])
            if return_feat:
                return sigmas, h, t_beta
            else:
                return sigmas
        else:
            if return_feat: return sigmas, h
            return sigmas

    def forward(self, x, d):
        """

        :param x: (N, 3) xyz in [-scale, scale]
        :param d: (N, 3) directions
        :return:
            sigmas: (N)
            rgb: (N, 3)
        """
        if "rffr" == self.dataset_name:
            # t
            t_sigmas, h, t_beta = self.density(x, return_feat=True)
            d = d / torch.norm(d, dim=1, keepdim=True)
            d = self.dir_encoder((d + 1) / 2)
            t_rgb = self.t_rgb(torch.cat([d, h], 1))

            # r
            r_rgb_sigmas = self.r_rgb(h)
            r_sigmas = TruncExp.apply(r_rgb_sigmas[:, -1:])
            r_rgb = TruncExp.apply(r_rgb_sigmas[:, :-1])

            return t_sigmas, t_rgb, t_beta, rearrange(r_sigmas, "n 1->n"), r_rgb
        else:
            sigmas, h = self.density(x, return_feat=True)
            d = d / torch.norm(d, dim=1, keepdim=True)
            d = self.dir_encoder((d + 1) / 2)
            rgb = self.rgb_net(torch.cat([d, h], 1))
            return sigmas, rgb

    @torch.no_grad()
    def get_all_cells(self):
        """
        从密度网格中获取所有单元格
        :return:
            cells 单元格
        """
        indices = vren.morton3D(self.grid_coords).long()  # grid_coords [2097152,3] 网格坐标（[1,128,128,128,3]）
        cells = [(indices, self.grid_coords)] * self.cascades
        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        对M均匀单元格和已占用单元格进行采样
        :param M:
        :param density_threshold: 密度阈值
        :return:
            cells 单元格
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c] > density_threshold)[:, 0]
            if len(indices2) > 0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64 ** 3):
        """
        密集网格初始化，用密度-1标记未被摄像机覆盖的单元格，训练开始前只执行一次
        :param K:  (3, 3) camera intrinsics
        :param poses:  (N, 3, 4) camera to world poses
        :param img_wh:  image width and height
        :param chunk: the chunk size to split the cells (to avoid OOM)
        :return:
        """
        N_cams = poses.shape[0]  # 100
        self.count_grid = torch.zeros_like(self.density_grid)  # density_grid [1, 128*128*128]
        w2c_R = poses[:, :3, :3].mT  # batch transpose w2c_R (N_cams, 3, 3) poses [100,3,4]
        w2c_T = -w2c_R @ poses[:, :3, 3:]  # (N_cams, 3, 1)
        cells = self.get_all_cells()  # 从密度网格中获取所有单元格
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i + chunk] / (self.grid_size - 1) * 2 - 1
                s = min(2 ** (c - 1), self.scale)
                half_grid_size = s / self.grid_size
                xyzs_w = (xyzs * (s - half_grid_size)).T  # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T  # (N_cams, 3, chunk)
                uvd = K @ xyzs_c  # (N_cams, 3, chunk)
                uv = uvd[:, :2] / uvd[:, 2:]  # (N_cams, 2, chunk)
                in_image = (uvd[:, 2] >= 0) & \
                           (uv[:, 0] >= 0) & (uv[:, 0] < img_wh[0]) & \
                           (uv[:, 1] >= 0) & (uv[:, 1] < img_wh[1])
                covered_by_cam = (uvd[:, 2] >= NEAR_DISTANCE) & in_image  # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i + chunk]] = \
                    count = covered_by_cam.sum(0) / N_cams

                too_near_to_cam = (uvd[:, 2] < NEAR_DISTANCE) & in_image  # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count > 0) & (~too_near_to_any_cam)
                self.density_grid[c, indices[i:i + chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        """
        更新单元格
        :param density_threshold: 密度阈值
        :param warmup: 预训练
        :param decay:  密度衰变
        :return:
        """
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup:  # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size ** 3 // 4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2 ** (c - 1), self.scale)
            half_grid_size = s / self.grid_size
            xyzs_w = (coords / (self.grid_size - 1) * 2 - 1) * (s - half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w) * 2 - 1) * half_grid_size
            density_grid_tmp[c, indices] = self.density(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay ** (1 / self.count_grid), 0.1, 0.95)

        self.density_grid = \
            torch.where(self.density_grid < 0,
                        self.density_grid,
                        torch.maximum(self.density_grid * decay, density_grid_tmp))
        mean_density = self.density_grid[self.density_grid > 0].mean().item()
        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)
