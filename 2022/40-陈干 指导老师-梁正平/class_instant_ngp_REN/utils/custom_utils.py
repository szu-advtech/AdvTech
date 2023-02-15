import torch
import torch.nn as nn
import vren
from torch.cuda.amp import custom_fwd, custom_bwd
from torch_scatter import segment_csr
from einops import rearrange


class RayAABBIntersector(torch.autograd.Function):
    """
    计算射线和轴向体素的交点
    Inputs:
        rays_o: (N_rays, 3) ray origins 射线原点
        rays_d: (N_rays, 3) ray directions 射线方向
        centers: (N_voxels, 3) voxel centers 立体像素中心
        half_sizes: (N_voxels, 3) voxel half sizes 立体像素一半边框
        max_hits: maximum number of intersected voxels to keep for one ray 为一条射线保留的相交体素的最大数目
                  (for a cubic scene, this is at most 3*N_voxels^(1/3)-2)

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray 每条射线的命中次数
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit)
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        return vren.ray_aabb_intersect(rays_o, rays_d, center, half_size, max_hits)


class RaySphereIntersector(torch.autograd.Function):
    """
    计算射线和球体的交点
    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_spheres, 3) sphere centers
        radii: (N_spheres, 3) radii
        max_hits: maximum number of intersected spheres to keep for one ray

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_sphere_idx: (N_rays, max_hits) hit sphere indices (-1 if no hit)
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, radii, max_hits):
        return vren.ray_sphere_intersect(rays_o, rays_d, center, radii, max_hits)


class RayMarcher(torch.autograd.Function):
    """
    移动射线得到样本点的位置和方向
    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) normalized ray directions
        hits_t: (N_rays, 2) near and far bounds from aabb intersection
        density_bitfield: (C*G**3//8)
        cascades: int
        scale: float
        exp_step_factor: the exponential factor to scale the steps
        grid_size: int
        max_samples: int
        mean_samples: int, mean total samples per batch

    Outputs:
        rays_a: (N_rays) ray_idx, start_idx, N_samples
        xyzs: (N, 3) sample positions
        dirs: (N, 3) sample view directions
        deltas: (N) dt for integration
        ts: (N) sample ts
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, hits_t,
                density_bitfield, cascades, scale, exp_step_factor,
                grid_size, max_samples):
        # noise to perturb the first sample of each ray 干扰每条射线的第一个样本的噪声
        noise = torch.rand_like(rays_o[:, 0])

        rays_a, xyzs, dirs, deltas, ts, counter = \
            vren.raymarching_train(
                rays_o, rays_d, hits_t,
                density_bitfield, cascades, scale,
                exp_step_factor, noise, grid_size, max_samples)

        total_samples = counter[0]  # total samples for all rays
        # remove redundant output
        xyzs = xyzs[:total_samples]
        dirs = dirs[:total_samples]
        deltas = deltas[:total_samples]
        ts = ts[:total_samples]

        ctx.save_for_backward(rays_a, ts)

        return rays_a, xyzs, dirs, deltas, ts, total_samples

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_drays_a, dL_dxyzs, dL_ddirs,
                 dL_ddeltas, dL_dts, dL_dtotal_samples):
        rays_a, ts = ctx.saved_tensors
        segments = torch.cat([rays_a[:, 1], rays_a[-1:, 1] + rays_a[-1:, 2]])
        dL_drays_o = segment_csr(dL_dxyzs, segments)
        dL_drays_d = \
            segment_csr(dL_dxyzs * rearrange(ts, 'n -> n 1') + dL_ddirs, segments)

        return dL_drays_o, dL_drays_d, None, None, None, None, None, None, None


class VolumeRenderer(torch.autograd.Function):
    """
    用于不同射线样本数量的射线的体绘制，仅在训练过程
    Inputs:
        sigmas: (N)
        rgbs: (N, 3)
        deltas: (N)
        ts: (N)
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
        T_threshold: float, stop the ray if the transmittance is below it

    Outputs:
        opacity: (N_rays)
        depth: (N_rays)
        depth_sq: (N_rays) expected value of squared distance
        rgb: (N_rays, 3)
    """

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, deltas, ts, rays_a, T_threshold):
        opacity, depth, depth_sq, rgb = \
            vren.composite_train_fw(sigmas, rgbs, deltas, ts,
                                    rays_a, T_threshold)
        ctx.save_for_backward(sigmas, rgbs, deltas, ts, rays_a,
                              opacity, depth, depth_sq, rgb)
        ctx.T_threshold = T_threshold
        return opacity, depth, depth_sq, rgb

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dopacity, dL_ddepth, dL_ddepth_sq, dL_drgb):
        sigmas, rgbs, deltas, ts, rays_a, \
        opacity, depth, depth_sq, rgb = ctx.saved_tensors
        dL_dsigmas, dL_drgbs = \
            vren.composite_train_bw(dL_dopacity, dL_ddepth, dL_ddepth_sq,
                                    dL_drgb, sigmas, rgbs, deltas, ts,
                                    rays_a,
                                    opacity, depth, depth_sq, rgb,
                                    ctx.T_threshold)
        return dL_dsigmas, dL_drgbs, None, None, None, None


class REN_VolumeRenderer(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, betas, sigmas, rgbs, deltas, ts, rays_a, T_threshold):
        beta, opacity, depth, depth_sq, rgb = \
            vren.REN_composite_train_fw(betas, sigmas, rgbs, deltas, ts,
                                        rays_a, T_threshold)
        ctx.save_for_backward(sigmas, rgbs, betas, deltas, ts, rays_a,
                              opacity, depth, depth_sq, rgb, beta)
        ctx.T_threshold = T_threshold
        return beta, opacity, depth, depth_sq, rgb

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dbeta, dL_dopacity, dL_ddepth, dL_ddepth_sq, dL_drgb):
        sigmas, rgbs, betas, deltas, ts, rays_a, \
        opacity, depth, depth_sq, rgb, beta = ctx.saved_tensors
        dL_dsigmas, dL_drgbs, dL_dbetas = \
            vren.REN_composite_train_bw(dL_dopacity, dL_ddepth, dL_ddepth_sq,
                                    dL_drgb, dL_dbeta, sigmas, rgbs, betas, deltas, ts,
                                    rays_a, opacity, depth, depth_sq, rgb, beta,ctx.T_threshold)
        return dL_dbetas,dL_dsigmas,dL_drgbs,None,None,None,None

class REN_depth_VolumeRenderer(torch.autograd.Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, deltas, ts, rays_a, T_threshold):
        opacity, depth, depth_sq = \
            vren.REN_composite_train_depth_fw(sigmas, deltas, ts, rays_a, T_threshold)
        ctx.save_for_backward(sigmas, deltas, ts, rays_a, opacity, depth, depth_sq)
        ctx.T_threshold = T_threshold
        return opacity, depth, depth_sq

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dopacity, dL_ddepth, dL_ddepth_sq):
        sigmas, deltas, ts, rays_a, opacity, depth, depth_sq = ctx.saved_tensors
        dL_dsigmas = \
            vren.REN_composite_train_depth_bw(dL_dopacity, dL_ddepth, dL_ddepth_sq,
                                              sigmas, deltas, ts, rays_a, opacity,
                                              depth, depth_sq, ctx.T_threshold)
        return dL_dsigmas[0], None, None, None, None


class VolumeRenderer_REN(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, sigmas, rgbs, deltas, ts, rays_a, t_threshold, beta):
        rays_n = rays_a.shape[0]
        rgb_out = torch.zeros(rays_n, 3, device=rays_a.device)
        depth = torch.zeros(rays_n, 1, device=rays_a.device)
        opacity = torch.zeros(rays_n, 1, device=rays_a.device)
        beta_out = None
        if beta is not None:
            beta_out = torch.zeros(rays_n, 1, device=rays_a.device)
        depth_b2f = torch.zeros(rays_n, 1, device=rays_a.device)

        sigmas_b2f = sigmas.flip(-1)
        deltas_b2f = depth_b2f.flip(-1)
        ts_b2f = ts.flip(-1)
        for ray_idx, value in enumerate(rays_a):
            samples = 0
            t = float(1.0)
            start_idx = value[1]
            while samples < value[2]:
                s = start_idx + samples
                a = float(1.0) - torch.exp(-sigmas[s] * deltas[s])
                w = a * t

                rgb_out[ray_idx][0] += w * rgbs[s][0]
                rgb_out[ray_idx][1] += w * rgbs[s][1]
                rgb_out[ray_idx][2] += w * rgbs[s][2]
                depth[ray_idx] += w * ts[s]
                if beta is not None:
                    beta_out[ray_idx] += w * beta
                opacity[ray_idx] += w
                t *= (float(1.0) - a)

                if t <= t_threshold:
                    break
                samples += 1

            samples_b2f = 0
            t_b2f = float(1.0)
            start_idx_b2f = value[1]
            while samples_b2f < value[2]:
                s_b2f = start_idx_b2f + samples_b2f
                a_b2f = float(1.0) - torch.exp(-sigmas_b2f[s_b2f] * deltas_b2f[s_b2f])
                w_b2f = a_b2f * t_b2f
                depth_b2f[ray_idx] += w_b2f * ts_b2f[s_b2f]
                t_b2f *= (float(1.0) - a_b2f)
                if t_b2f <= t_threshold:
                    break
                samples_b2f += 1

        return opacity, depth, rgb_out, depth_b2f, beta_out

    @torch.no_grad()
    def volumeRenderer_REN_test(sigmas, rgbs, deltas, ts,
                                alive_indices, t_threshold,
                                n_eff_samples, beat,
                                opacity, depth, rgb, beat_out):
        with torch.no_grad:
            for n in range(len(alive_indices)):
                if n_eff_samples[n] == 0:
                    alive_indices[n] = -1
                    continue
                r = alive_indices[n]
                s = 0
                t = 1 - opacity[r]
                while s < n_eff_samples[n]:
                    a = float(1.0) - torch.exp(-sigmas[n][s] * deltas[n][s])
                    w = a * t
                    rgb[r][0] += w * rgbs[n][s][0]
                    rgb[r][1] += w * rgbs[n][s][1]
                    rgb[r][2] += w * rgbs[n][s][2]
                    depth[r] += w * ts[n][s]
                    if beat is not None:
                        beat_out[r] += w * beat[n][s]
                    opacity[r] += w
                    t *= float(1.0) - a
                    if t < t_threshold:
                        alive_indices[n] = -1
                        continue
                    s += 1


class TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))
