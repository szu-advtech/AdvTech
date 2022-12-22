import torch
import numpy as np
from kornia import create_meshgrid
from einops import rearrange


@torch.cuda.amp.autocast(dtype=torch.float32)
def get_ray_directions(H, W, K, device='cpu', random=False, return_uv=False, flatten=True):
    """
    获取相机坐标中像素的光线方向
    :param H: 图像高
    :param W: 图像宽
    :param K: (3, 3) 相加内参
    :param device: cpu gpu tpu
    :param random: 光线是否随机穿过像素内部
    :param return_uv: 是否返回图像uv坐标
    :param flatten: 是否压缩
    :return:
         directions: (H, W, 3) or (H*W, 3) 相机坐标中像素的光线方向
         uv: (H, W, 2) or (H*W, 2) 图像uv坐标
    """
    grid = create_meshgrid(H, W, False, device=device)[0]  # (H, W, 2) 为图像生成一个坐标网格
    u, v = grid.unbind(-1)  # U:[800，800] V:[800，800]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]  # fx,fy,w/2,h/2
    if random:
        directions = \
            torch.stack([(u - cx + torch.rand_like(u)) / fx,
                         (v - cy + torch.rand_like(v)) / fy,
                         torch.ones_like(u)], -1)
    else:  # 光线从像素中心穿过
        directions = \
            torch.stack([(u - cx + 0.5) / fx, (v - cy + 0.5) / fy, torch.ones_like(u)], -1)
    if flatten:
        directions = directions.reshape(-1, 3)
        grid = grid.reshape(-1, 2)
    if return_uv:
        return directions, grid
    return directions


@torch.cuda.amp.autocast(dtype=torch.float32)
def get_rays(directions, c2w):
    """
    获取一个图像中像素在世界坐标中的射线原点和方向
    :param directions: 相机坐标中像素的光线方向
    :param c2w: 相机坐标转换世界坐标矩阵
    :return:
         rays_o: (N, 3), 射线原点
         rays_d: (N, 3), 射线方向
    """
    # 从相机坐标到世界坐标
    if c2w.ndim == 2:
        rays_d = directions @ c2w[:, :3].T
    else:
        rays_d = rearrange(directions, 'n c -> n 1 c') @ c2w[..., :3].mT
        rays_d = rearrange(rays_d, 'n 1 c -> n c')
    # 图像上所有光线的原点都是摄像机在世界坐标中的原点
    rays_o = c2w[..., 3].expand_as(rays_d)
    return rays_o, rays_d


@torch.cuda.amp.autocast(dtype=torch.float32)
def axisangle_to_R(v):
    """
    将轴角向量转换为旋转矩阵
    :param v: (B, 3)
    :return: (B, 3, 3)
    """
    zero = torch.zeros_like(v[:, :1])  # (B, 1)
    skew_v0 = torch.cat([zero, -v[:, 2:3], v[:, 1:2]], 1)  # (B, 3)
    skew_v1 = torch.cat([v[:, 2:3], zero, -v[:, 0:1]], 1)
    skew_v2 = torch.cat([-v[:, 1:2], v[:, 0:1], zero], 1)
    skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=1)  # (B, 3, 3)

    norm_v = rearrange(torch.norm(v, dim=1) + 1e-7, 'b -> b 1 1')
    eye = torch.eye(3, device=v.device)
    R = eye + (torch.sin(norm_v) / norm_v) * skew_v + \
        ((1 - torch.cos(norm_v)) / norm_v ** 2) * (skew_v @ skew_v)
    return R


def normalize(v):
    """
    标准化一个向量
    :param v:
    :return:
    """
    return v / np.linalg.norm(v)


def average_poses(poses, pts3d=None):
    """
    计算平均pose，然后用来居中所有pose
    :param poses: (N_images, 3, 4)
    :param pts3d: (N, 3)
    :return:
        pose_avg: (3, 4) 平均pose

    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of 3d point cloud (if None, center of cameras).
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    """
    # 1. Compute the center
    if pts3d is not None:
        center = pts3d.mean(0)
    else:
        center = poses[..., 3].mean(0)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(y_, z))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(z, x)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


# LLFF数据预处理
def center_poses(poses, pts3d=None):
    """
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
        pts3d: (N, 3) reconstructed point cloud

    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pts3d_centered: (N, 3) centered point cloud
    """

    pose_avg = average_poses(poses, pts3d)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[:3] = pose_avg  # convert to homogeneous coordinate for faster computation
    # by simply adding 0, 0, 0, 1 as the last row
    pose_avg_inv = np.linalg.inv(pose_avg_homo)
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = \
        np.concatenate([poses, last_row], 1)  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = pose_avg_inv @ poses_homo  # (N_images, 4, 4)
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    if pts3d is not None:
        pts3d_centered = pts3d @ pose_avg_inv[:, :3].T + pose_avg_inv[:, 3:].T
        return poses_centered, pts3d_centered

    return poses_centered


def create_spheric_poses(radius, mean_h, n_poses=120):
    """
    围绕z轴创建圆形姿势
    :param radius: 圆的(复数)高和半径
    :param mean_h: 平均相机高度
    :param n_poses:
    :return: spheric_poses (n_poses, 3, 4) 圆形路径的姿势
    """

    def spheric_pose(theta, phi, radius):
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 2 * mean_h],
            [0, 0, 1, -t]
        ])

        rot_phi = lambda phi: np.array([
            [1, 0, 0],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi), np.cos(phi)]
        ])

        rot_theta = lambda th: np.array([
            [np.cos(th), 0, -np.sin(th)],
            [0, 1, 0],
            [np.sin(th), 0, np.cos(th)]
        ])

        c2w = rot_theta(theta) @ rot_phi(phi) @ trans_t(radius)
        c2w = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]]) @ c2w
        return c2w

    spheric_poses = []
    for th in np.linspace(0, 2 * np.pi, n_poses + 1)[:-1]:
        spheric_poses += [spheric_pose(th, -np.pi / 12, radius)]
    return np.stack(spheric_poses, 0)
