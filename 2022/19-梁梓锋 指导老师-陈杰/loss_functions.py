from __future__ import division
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from inverse_warp import inverse_warp

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        k = 7
        self.mu_x_pool = nn.AvgPool2d(k, 1)
        self.mu_y_pool = nn.AvgPool2d(k, 1)
        self.sig_x_pool = nn.AvgPool2d(k, 1)
        self.sig_y_pool = nn.AvgPool2d(k, 1)
        self.sig_xy_pool = nn.AvgPool2d(k, 1)

        self.refl = nn.ReflectionPad2d(k//2)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * \
            (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
compute_ssim_loss = SSIM().to(device)

# 光度重建误差，对应论文中的Lvs
# 需要target_img、source_imgs、相机内参、target_img的深度、可解释性掩模、target到source的位姿（t->t-1,t->t+1）
def photometric_reconstruction_loss(tgt_img, ref_imgs, intrinsics,
                                    depth, explainability_mask, pose,
                                    rotation_mode='euler', padding_mode='zeros'):
    def one_scale(depth, explainability_mask):
        assert(explainability_mask is None or depth.size()[2:] == explainability_mask.size()[2:])
        assert(pose.size(1) == len(ref_imgs))

        reconstruction_loss = 0 # 重建损失初始化为0
        b, _, h, w = depth.size() # batch，_, h, w 深度图的size，通过depth_net获得的深度图，tgt_img的深度图
        downscale = tgt_img.size(2)/h # 缩放比例，图片和深度图可能不一致

        tgt_img_scaled = F.interpolate(tgt_img, (h, w), mode='area') # 上采样或者下采样，使图片和深度图的大小一致
        ref_imgs_scaled = [F.interpolate(ref_img, (h, w), mode='area') for ref_img in ref_imgs] # 使图片和深度图的大小一致
        # 内参缩放？为什么这样缩放？batch_size为4，对每一个样本的内参的前两行，也就是fx、fy、cx、cy都除以缩放比例即可，第三行第三列为1，不变
        intrinsics_scaled = torch.cat((intrinsics[:, 0:2]/downscale, intrinsics[:, 2:]), dim=1)

        warped_imgs = []
        diff_maps = []

        for i, ref_img in enumerate(ref_imgs_scaled):
            current_pose = pose[:, i]
            # 把握住它的本质和光流法相差不大，就是跟踪物体的像素点在相邻的两幅图像中的光度应该是相差不大的（光度一致性）
            # 所以将tgt_img的所有像素坐标通过K、Depth、T就可以得到这些坐标在ref_img下的坐标
            # 所谓的插值，就是输入一副原图，然后输入新图上每个像素的坐标（在原图上的坐标），根据这些坐标在原图上插值得到像素值，作为新图的像素
            # 所以坐标不同，就可以实现物体的形变，所以称为warp扭曲
            ref_img_warped, valid_points = inverse_warp(ref_img, depth[:,0], current_pose,
                                                        intrinsics_scaled,
                                                        rotation_mode, padding_mode)
            diff = (tgt_img_scaled - ref_img_warped) * valid_points.unsqueeze(1).float() # tgt_img的光度值减去ref_img_warped的光度值
            ssim_loss = compute_ssim_loss(tgt_img_scaled, ref_img_warped)   # 加入ssim结构相似性
            diff = (0.85 * diff + 0.15 * ssim_loss)

            if explainability_mask is not None:
                diff = diff * explainability_mask[:,i:i+1].expand_as(diff)

            reconstruction_loss += diff.abs().mean() # 全部误差加起来
            assert((reconstruction_loss == reconstruction_loss).item() == 1)

            warped_imgs.append(ref_img_warped[0])
            diff_maps.append(diff[0])

        return reconstruction_loss, warped_imgs, diff_maps

    # 分了不同尺度，分不同尺度的原因就是解决梯度的局部性（弱纹理，正确的像素与估计相距很远的情况）
    warped_results, diff_results = [], []
    if type(explainability_mask) not in [tuple, list]:
        explainability_mask = [explainability_mask] # Es，可解释性掩模
    if type(depth) not in [list, tuple]:
        depth = [depth] # 深度

    total_loss = 0
    for d, mask in zip(depth, explainability_mask):
        loss, warped, diff = one_scale(d, mask)
        total_loss += loss
        warped_results.append(warped)
        diff_results.append(diff)
    return total_loss, warped_results, diff_results


def explainability_loss(mask):
    if type(mask) not in [tuple, list]:
        mask = [mask]
    loss = 0
    for mask_scaled in mask:
        ones_var = torch.ones_like(mask_scaled)
        loss += nn.functional.binary_cross_entropy(mask_scaled, ones_var)
    return loss

# 平滑损失
def smooth_loss(pred_map):
    def gradient(pred):
        # 求梯度图
        # y方向梯度就是沿y方向将后一个像素减去前一个像素，最后一个像素没法算
        D_dy = pred[:, :, 1:] - pred[:, :, :-1]
        # x同理
        D_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        return D_dx, D_dy

    if type(pred_map) not in [tuple, list]:
        pred_map = [pred_map]

    loss = 0
    weight = 1.

    for scaled_map in pred_map:
        dx, dy = gradient(scaled_map)
        dx2, dxdy = gradient(dx) # 对dx再次求导
        dydx, dy2 = gradient(dy) # 对dy再次求导
        # 所以平滑损失是最小化二阶导的L1 loss
        loss += (dx2.abs().mean() + dxdy.abs().mean() + dydx.abs().mean() + dy2.abs().mean())*weight
        weight /= 2.3  # don't ask me why it works better
    return loss


@torch.no_grad()
def compute_depth_errors(gt, pred, crop=True):
    abs_diff, abs_rel, sq_rel, a1, a2, a3 = 0,0,0,0,0,0
    batch_size = gt.size(0)

    '''
    crop used by Garg ECCV16 to reprocude Eigen NIPS14 results
    construct a mask of False values, with the same size as target
    and then set to True values inside the crop
    '''
    if crop:
        crop_mask = gt[0] != gt[0]
        y1,y2 = int(0.40810811 * gt.size(1)), int(0.99189189 * gt.size(1))
        x1,x2 = int(0.03594771 * gt.size(2)), int(0.96405229 * gt.size(2))
        crop_mask[y1:y2,x1:x2] = 1
    skipped = 0
    for current_gt, current_pred in zip(gt, pred):
        valid = (current_gt > 0) & (current_gt < 80)
        if crop:
            valid = valid & crop_mask
        if valid.sum() == 0:
            continue

        valid_gt = current_gt[valid]
        valid_pred = current_pred[valid].clamp(1e-3, 80)

        valid_pred = valid_pred * torch.median(valid_gt)/torch.median(valid_pred)

        thresh = torch.max((valid_gt / valid_pred), (valid_pred / valid_gt))
        a1 += (thresh < 1.25).float().mean()
        a2 += (thresh < 1.25 ** 2).float().mean()
        a3 += (thresh < 1.25 ** 3).float().mean()

        abs_diff += torch.mean(torch.abs(valid_gt - valid_pred))
        abs_rel += torch.mean(torch.abs(valid_gt - valid_pred) / valid_gt)

        sq_rel += torch.mean(((valid_gt - valid_pred)**2) / valid_gt)
    if skipped == batch_size:
        return None

    return [metric.item() / (batch_size - skipped) for metric in [abs_diff, abs_rel, sq_rel, a1, a2, a3]]


@torch.no_grad()
def compute_pose_errors(gt, pred):
    RE = 0
    for (current_gt, current_pred) in zip(gt, pred):
        snippet_length = current_gt.shape[0]
        scale_factor = torch.sum(current_gt[..., -1] * current_pred[..., -1]) / torch.sum(current_pred[..., -1] ** 2)
        ATE = torch.norm((current_gt[..., -1] - scale_factor * current_pred[..., -1]).reshape(-1)).cpu().numpy()
        R = current_gt[..., :3] @ current_pred[..., :3].transpose(-2, -1)
        for gt_pose, pred_pose in zip(current_gt, current_pred):
            R = (gt_pose[:, :3] @ torch.inverse(pred_pose[:, :3])).cpu().numpy()
            s = np.linalg.norm([R[0, 1]-R[1, 0],
                                R[1, 2]-R[2, 1],
                                R[0, 2]-R[2, 0]])
            c = np.trace(R) - 1
            RE += np.arctan2(s, c)
    return [ATE/snippet_length, RE/snippet_length]
