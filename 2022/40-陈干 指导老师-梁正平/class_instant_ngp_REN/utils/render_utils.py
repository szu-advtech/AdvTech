import torch

import models.networks
from .custom_utils import \
    RayAABBIntersector, RayMarcher, VolumeRenderer, REN_VolumeRenderer, REN_depth_VolumeRenderer
from einops import rearrange, repeat
import vren
import cv2
import numpy as np

MAX_SAMPLES = 1024
NEAR_DISTANCE = 0.01


def render(model, rays_o, rays_d, **kwargs):
    """
    渲染
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    rays_o = rays_o.contiguous();
    rays_d = rays_d.contiguous()
    """
        hits_cnt: (N_rays) number of hits for each ray 每条射线的命中次数
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit) 
    """
    # center[0,0,0] half_size 0.5
    # 射线与密集网格相交
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0] >= 0) & (hits_t[:, 0, 0] < NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test
    else:
        render_func = __render_rays_train
    # 渲染
    results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        results[k] = v
    return results


@torch.no_grad()
def __render_rays_test(model, rays_o, rays_d, hits_t, **kwargs):
    """
    渲染测试光线

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device
    opacity = torch.zeros(N_rays, device=device)
    depth = torch.zeros(N_rays, device=device)
    rgb = torch.zeros(N_rays, 3, device=device)

    samples = total_samples = 0
    alive_indices = torch.arange(N_rays, device=device)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor == 0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive == 0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays // N_alive, 64), min_samples)
        samples += N_samples

        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)
        total_samples += N_eff_samples.sum()
        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs == 0, dim=1)
        if valid_mask.sum() == 0: break

        sigmas = torch.zeros(len(xyzs), device=device)
        rgbs = torch.zeros(len(xyzs), 3, device=device)
        _sigmas, _rgbs = model(xyzs[valid_mask], dirs[valid_mask])
        sigmas[valid_mask], rgbs[valid_mask] = _sigmas.float(), _rgbs.float()
        sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)

        vren.composite_test_fw(
            sigmas, rgbs, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity, depth, rgb)
        alive_indices = alive_indices[alive_indices >= 0]  # remove converged rays

    results['opacity'] = opacity
    results['depth'] = depth
    results['rgb'] = rgb
    results['total_samples'] = total_samples  # total samples for all rays

    if exp_step_factor == 0:  # synthetic
        rgb_bg = torch.ones(3, device=device)
    else:  # real
        rgb_bg = torch.zeros(3, device=device)
    results['rgb'] += rgb_bg * rearrange(1 - opacity, 'n -> n 1')

    return results


@torch.cuda.amp.autocast()
def __render_rays_train(model, rays_o, rays_d, hits_t, **kwargs):
    """
    渲染训练射线
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
       沿着射线的方向移动，查询@density_bitfield以跳过空白，并获得有效的采样点(在有对象的地方)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
       推断这些位置的NN并查看方向以获得属性(目前是sigma和rgb)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
       使用体渲染来合并结果(前后合成，如果光线的透过率低于阈值，则提前停止光线)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    results = {}
    # 沿着射线的方向移动，查询@density_bitfield以跳过空白，并获得有效的采样点(在有对象的地方)
    rays_a, xyzs, dirs, deltas, ts, total_samples = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
            model.cascades, model.scale,
            exp_step_factor, model.grid_size, MAX_SAMPLES)
    results['total_samples'] = total_samples

    # 推断这些位置的NN并查看方向以获得属性(目前是sigma和rgb)
    for k, v in kwargs.items():  # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
    # 从模型获取预测值
    sigmas, rgbs = model(xyzs, dirs)

    results['opacity'], results['depth'], _, results['rgb'] = \
        VolumeRenderer.apply(sigmas, rgbs.contiguous(), deltas, ts,
                             rays_a, kwargs.get('T_threshold', 1e-4))

    # 使用体渲染来合并结果(前后合成，如果光线的透过率低于阈值，则提前停止光线)
    if exp_step_factor == 0:  # synthetic
        rgb_bg = torch.ones(3, device=rays_o.device)
    else:  # real
        if kwargs.get('random_bg', False):
            rgb_bg = torch.rand(3, device=rays_o.device)
        else:
            rgb_bg = torch.zeros(3, device=rays_o.device)
    results['rgb'] = results['rgb'] + \
                     rgb_bg * rearrange(1 - results['opacity'], 'n -> n 1')

    return results


def render_REN(model, rays_o, rays_d, **kwargs):
    """
    渲染
    Render rays by
    1. Compute the intersection of the rays with the scene bounding box
    2. Follow the process in @render_func (different for train/test)

    Inputs:
        model: NGP
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions

    Outputs:
        result: dictionary containing final rgb and depth
    """
    rays_o = rays_o.contiguous();
    rays_d = rays_d.contiguous()
    """
        hits_cnt: (N_rays) number of hits for each ray 每条射线的命中次数
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit) 
    """
    # center[0,0,0] half_size 0.5
    # 射线与密集网格相交
    _, hits_t, _ = \
        RayAABBIntersector.apply(rays_o, rays_d, model.center, model.half_size, 1)
    hits_t[(hits_t[:, 0, 0] >= 0) & (hits_t[:, 0, 0] < NEAR_DISTANCE), 0, 0] = NEAR_DISTANCE

    if kwargs.get('test_time', False):
        render_func = __render_rays_test_REN
    else:
        render_func = __render_rays_train_REN
    # 渲染
    t_results, r_results = render_func(model, rays_o, rays_d, hits_t, **kwargs)
    for k, v in t_results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        t_results[k] = v
    for k, v in r_results.items():
        if kwargs.get('to_cpu', False):
            v = v.cpu()
            if kwargs.get('to_numpy', False):
                v = v.numpy()
        r_results[k] = v
    return t_results, r_results


@torch.no_grad()
def __render_rays_test_REN(model, rays_o, rays_d, hits_t, **kwargs):
    """
    渲染测试光线

    while (a ray hasn't converged)
        1. Move each ray to its next occupied @N_samples (initially 1) samples
           and evaluate the properties (sigmas, rgbs) there
        2. Composite the result to output; if a ray has transmittance lower
           than a threshold, mark this ray as converged and stop marching it.
           When more rays are dead, we can increase the number of samples
           of each marching (the variable @N_samples)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    t_results = {}
    r_results = {}

    # output tensors to be filled in
    N_rays = len(rays_o)
    device = rays_o.device

    opacity_t = torch.zeros(N_rays, device=device, requires_grad=False)
    depth_t = torch.zeros(N_rays, device=device, requires_grad=False)
    rgb_t = torch.zeros(N_rays, 3, device=device, requires_grad=False)

    beta = torch.zeros(N_rays, device=device, requires_grad=False)

    opacity_r = torch.zeros(N_rays, device=device, requires_grad=False)
    depth_r = torch.zeros(N_rays, device=device, requires_grad=False)
    rgb_r = torch.zeros(N_rays, 3, device=device, requires_grad=False)

    samples = total_samples = 0

    alive_indices = torch.arange(N_rays, device=device, requires_grad=False)
    # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
    # otherwise, 4 is more efficient empirically
    min_samples = 1 if exp_step_factor == 0 else 4

    while samples < kwargs.get('max_samples', MAX_SAMPLES):
        N_alive = len(alive_indices)
        if N_alive == 0: break

        # the number of samples to add on each ray
        N_samples = max(min(N_rays // N_alive, 64), min_samples)
        samples += N_samples
        xyzs, dirs, deltas, ts, N_eff_samples = \
            vren.raymarching_test(rays_o, rays_d, hits_t[:, 0], alive_indices,
                                  model.density_bitfield, model.cascades,
                                  model.scale, exp_step_factor,
                                  model.grid_size, MAX_SAMPLES, N_samples)

        total_samples += N_eff_samples.sum()


        xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
        dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
        valid_mask = ~torch.all(dirs == 0, dim=1)
        if valid_mask.sum() == 0: break

        t_sigmas, t_rgb, t_beta, r_sigmas, r_rgb = model(xyzs[valid_mask], dirs[valid_mask])

        sigmas_t = torch.zeros(len(xyzs), device=device, requires_grad=False)
        rgbs_t = torch.zeros(len(xyzs), 3, device=device, requires_grad=False)
        betas_t = torch.zeros(len(xyzs), device=device, requires_grad=False)
        sigmas_t[valid_mask], rgbs_t[valid_mask], betas_t[valid_mask] = t_sigmas.float(), t_rgb.float(), t_beta.float()
        sigmas_t = rearrange(sigmas_t, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs_t = rearrange(rgbs_t, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        betas_t = rearrange(betas_t, '(n1 n2) -> n1 n2', n2=N_samples)
        vren.REN_composite_test_fw(
            sigmas_t, rgbs_t, betas_t, deltas, ts,
            hits_t[:, 0], alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity_t, depth_t, rgb_t, beta)

        r_alive_indices = alive_indices


        sigmas_r = torch.zeros(len(xyzs), device=device, requires_grad=False)
        rgbs_r = torch.zeros(len(xyzs), 3, device=device, requires_grad=False)
        sigmas_r[valid_mask], rgbs_r[valid_mask] = r_sigmas.float(), r_rgb.float()
        sigmas_r = rearrange(sigmas_r, '(n1 n2) -> n1 n2', n2=N_samples)
        rgbs_r = rearrange(rgbs_r, '(n1 n2) c -> n1 n2 c', n2=N_samples)
        vren.composite_test_fw(
            sigmas_r, rgbs_r, deltas, ts,
            hits_t[:, 0], r_alive_indices, kwargs.get('T_threshold', 1e-4),
            N_eff_samples, opacity_r, depth_r, rgb_r)

        alive_indices = alive_indices[alive_indices >= 0]


    t_results['opacity'] = opacity_t
    t_results['depth'] = depth_t
    t_results['beta'] = rearrange(beta, "n->n 1")
    t_results['rgb'] = rgb_t
    t_results['total_samples'] = total_samples  # total samples for all rays

    r_results['opacity'] = opacity_r
    r_results['depth'] = depth_r
    r_results['rgb'] = rgb_r

    if exp_step_factor == 0:  # synthetic
        rgb_bg = torch.ones(3, device=device, requires_grad=False)
    else:  # real
        rgb_bg = torch.zeros(3, device=device, requires_grad=False)

    t_results['rgb'] = t_results['rgb'] + \
                       rgb_bg * (1 - rearrange(t_results['opacity'], "n -> n 1"))
    r_results['rgb'] = r_results['rgb'] + \
                       rgb_bg * (1 - rearrange(r_results['opacity'], "n -> n 1"))
    return t_results, r_results


@torch.cuda.amp.autocast()
def __render_rays_train_REN(model, rays_o, rays_d, hits_t, **kwargs):
    """
    渲染训练射线
    Render rays by
    1. March the rays along their directions, querying @density_bitfield
       to skip empty space, and get the effective sample points (where
       there is object)
       沿着射线的方向移动，查询@density_bitfield以跳过空白，并获得有效的采样点(在有对象的地方)
    2. Infer the NN at these positions and view directions to get properties
       (currently sigmas and rgbs)
       推断这些位置的NN并查看方向以获得属性(目前是sigma和rgb)
    3. Use volume rendering to combine the result (front to back compositing
       and early stop the ray if its transmittance is below a threshold)
       使用体渲染来合并结果(前后合成，如果光线的透过率低于阈值，则提前停止光线)
    """
    exp_step_factor = kwargs.get('exp_step_factor', 0.)
    t_results = {}
    r_results = {}

    # 沿着射线的方向移动，查询@density_bitfield以跳过空白，并获得有效的采样点(在有对象的地方)
    rays_a, xyzs, dirs, deltas, ts, total_samples = \
        RayMarcher.apply(
            rays_o, rays_d, hits_t[:, 0], model.density_bitfield,
            model.cascades, model.scale,
            exp_step_factor, model.grid_size, MAX_SAMPLES)
    t_results['total_samples'] = total_samples
    # 推断这些位置的NN并查看方向以获得属性(目前是sigma和rgb)
    for k, v in kwargs.items():  # supply additional inputs, repeated per ray
        if isinstance(v, torch.Tensor):
            kwargs[k] = torch.repeat_interleave(v[rays_a[:, 0]], rays_a[:, 2], 0)
    # 从模型获取预测值
    t_sigmas, t_rgbs, betas, r_sigmas, r_rgbs = model(xyzs, dirs)

    t_results["beta"], t_results['opacity'], t_results['depth'], _, t_results['rgb'] = \
        REN_VolumeRenderer.apply(betas, t_sigmas, t_rgbs.contiguous(), deltas, ts,
                                 rays_a, kwargs.get('T_threshold', 1e-4))

    t_results['opacity_b2f'], t_results['depth_b2f'], _ = \
        REN_depth_VolumeRenderer.apply(t_sigmas.flip(-1), deltas.flip(-1), ts.flip(-1), rays_a, kwargs.get('T_threshold', 1e-4))

    r_results['opacity'], r_results['depth'], _, r_results['rgb'] = \
        VolumeRenderer.apply(r_sigmas, r_rgbs.contiguous(), deltas, ts,
                             rays_a, kwargs.get('T_threshold', 1e-4))

    r_results['opacity_b2f'], r_results['depth_b2f'], _ = \
        REN_depth_VolumeRenderer.apply(r_sigmas.flip(-1), deltas.flip(-1), ts.flip(-1), rays_a, kwargs.get('T_threshold', 1e-4))

    # 使用体渲染来合并结果(前后合成，如果光线的透过率低于阈值，则提前停止光线)
    if exp_step_factor == 0:  # synthetic
        rgb_bg = torch.ones(3, device=rays_o.device)
    else:  # real
        if kwargs.get('random_bg', False):
            rgb_bg = torch.rand(3, device=rays_o.device)
        else:
            rgb_bg = torch.zeros(3, device=rays_o.device)

    t_results["beta"] = rearrange(t_results["beta"], "n-> n 1")

    t_results['rgb'] = t_results['rgb'] + \
                       rgb_bg * (1 - rearrange(t_results['opacity'], "n -> n 1"))
    r_results['rgb'] = r_results['rgb'] + \
                       rgb_bg * (1 - rearrange(r_results['opacity'], "n -> n 1"))

    return t_results, r_results
