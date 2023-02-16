import torch
import torch.nn as nn
import torch.nn.functional as F

from encoding import get_encoder
from activation import trunc_exp
from .renderer import NeRFRenderer
from gridencoder import GridEncoder
from .utils import custom_meshgrid
import numpy as np
import tinycudann as tcnn
from copy import deepcopy
import raymarching


class NeRFNetwork(NeRFRenderer):
    def __init__(self,
                 encoding="hashgrid",
                 encoding_dir="sphere_harmonics",
                 encoding_bg="hashgrid",
                 num_layers=2,
                 hidden_dim=64,
                 geo_feat_dim=15,
                 num_layers_color=3,
                 hidden_dim_color=64,
                 num_layers_bg=2,
                 hidden_dim_bg=64,
                 bound=1,
                 device="cuda:0",
                 **kwargs,
                 ):
        super().__init__(bound, **kwargs)

        # sigma network
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.geo_feat_dim = geo_feat_dim
        self.lambda1 = 1
        self.lambda2 = 1
        self.lambda3 = 0.1
        self.device = device

        self.encoder, self.in_dim = get_encoder(encoding, \
                input_dim=3,num_levels=16, level_dim=4, \
                base_resolution=16, log2_hashmap_size=14,\
                gridtype='hash', align_corners=False,\
                desired_resolution=1024 * bound)

        std = 1e-4
        self.H = [] 
        for i in range(1 + 46):
            hash_tables = torch.zeros_like(self.encoder.embeddings.data, requires_grad=True, device=self.device)
            hash = nn.Parameter(hash_tables) # need to be optimized
            hash.data.uniform_(-std, std)
            self.H.append(hash) # store the paramaters objects(share id )

        self.H = nn.ParameterList(self.H)

        if self.cuda_ray:
            self.density_grid_list = []
            for i in range(1 + 46):
                dg = deepcopy(self.density_grid)
                self.density_grid_list.append(dg)
                # self.register_buffer('density_grid_list', density_grid_list)

        sigma_net = []
        for l in range(num_layers):
            if l == 0:
                in_dim = self.in_dim
            else:
                in_dim = hidden_dim
            
            if l == num_layers - 1:
                out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
            else:
                out_dim = hidden_dim
            
            sigma_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.sigma_net = nn.ModuleList(sigma_net)

        # color network
        self.num_layers_color = num_layers_color        
        self.hidden_dim_color = hidden_dim_color
        self.encoder_dir, self.in_dim_dir = get_encoder(encoding_dir)
        
        color_net =  []
        for l in range(num_layers_color):
            if l == 0:
                in_dim = self.in_dim_dir + self.geo_feat_dim
            else:
                in_dim = hidden_dim_color
            
            if l == num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))

        self.color_net = nn.ModuleList(color_net)

        # background network
        if self.bg_radius > 0:
            self.num_layers_bg = num_layers_bg        
            self.hidden_dim_bg = hidden_dim_bg
            self.encoder_bg, self.in_dim_bg = get_encoder(encoding_bg, input_dim=2, num_levels=4, log2_hashmap_size=19, desired_resolution=2048) # much smaller hashgrid 
            
            bg_net = []
            for l in range(num_layers_bg):
                if l == 0:
                    in_dim = self.in_dim_bg + self.in_dim_dir
                else:
                    in_dim = hidden_dim_bg
                
                if l == num_layers_bg - 1:
                    out_dim = 3 # 3 rgb
                else:
                    out_dim = hidden_dim_bg
                
                bg_net.append(nn.Linear(in_dim, out_dim, bias=False))

            self.bg_net = nn.ModuleList(bg_net)
        else:
            self.bg_net = None

    @torch.no_grad()
    def mark_untrained_grid(self, poses, intrinsic, S=64):
        # poses: [B, 4, 4]
        # intrinsic: [3, 3]

        if not self.cuda_ray:
            return
        
        if isinstance(poses, np.ndarray):
            poses = torch.from_numpy(poses)

        B = poses.shape[0]
        
        fx, fy, cx, cy = intrinsic
        
        X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
        Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

        count = torch.zeros_like(self.density_grid)
        poses = poses.to(count.device)

        # 5-level loop, forgive me...

        for xs in X:
            for ys in Y:
                for zs in Z:
                    
                    # construct points
                    xx, yy, zz = custom_meshgrid(xs, ys, zs)
                    coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                    indices = raymarching.morton3D(coords).long() # [N]
                    world_xyzs = (2 * coords.float() / (self.grid_size - 1) - 1).unsqueeze(0) # [1, N, 3] in [-1, 1]

                    # cascading
                    for cas in range(self.cascade):
                        bound = min(2 ** cas, self.bound)
                        half_grid_size = bound / self.grid_size
                        # scale to current cascade's resolution
                        cas_world_xyzs = world_xyzs * (bound - half_grid_size)

                        # split batch to avoid OOM
                        head = 0
                        while head < B:
                            tail = min(head + S, B)

                            # world2cam transform (poses is c2w, so we need to transpose it. Another transpose is needed for batched matmul, so the final form is without transpose.)
                            cam_xyzs = cas_world_xyzs - poses[head:tail, :3, 3].unsqueeze(1)
                            cam_xyzs = cam_xyzs @ poses[head:tail, :3, :3] # [S, N, 3]
                            
                            # query if point is covered by any camera
                            mask_z = cam_xyzs[:, :, 2] > 0 # [S, N]
                            mask_x = torch.abs(cam_xyzs[:, :, 0]) < cx / fx * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask_y = torch.abs(cam_xyzs[:, :, 1]) < cy / fy * cam_xyzs[:, :, 2] + half_grid_size * 2
                            mask = (mask_z & mask_x & mask_y).sum(0).reshape(-1) # [N]

                            # update count 
                            count[cas, indices] += mask
                            head += S
    
        # mark untrained grid as -1
        self.density_grid[count == 0] = -1
        for i in range(47):
            self.density_grid_list[i][count == 0] = -1

        print(f'[mark untrained grid] {(count == 0).sum()} from {self.grid_size ** 3 * self.cascade}')


    @torch.no_grad()
    def update_extra_state(self, max_weight, decay=0.95, S=128, bs_weight=None):
        # weight
        dg_list = []
        for i in range(1, 47):
            self.update_extra_state_n(idx=i, max_weight=max_weight, decay=0.95, S=128, bs_weight=None)
            dg_list.append(self.density_grid_list[i])

        self.density_grid, _ = torch.stack(dg_list).max(axis=0)
        self.mean_density = torch.mean(self.density_grid[self.density_grid > 0]).item() # do not count -1 regions
        density_thresh = min(self.mean_density, self.density_thresh)
        self.density_bitfield = raymarching.packbits(self.density_grid, density_thresh, self.density_bitfield)

        self.iter_density += 1
        ### update step counter
        total_step = min(16, self.local_step)
        if total_step > 0:
            self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
        self.local_step = 0

    @torch.no_grad()
    def update_extra_state_n(self, idx=None, max_weight=None, decay=0.95, S=128, bs_weight=None):
        # call before each epoch to update extra states.

        if not self.cuda_ray:
            return 
        
        if idx is not None:
            density_grid = self.density_grid_list[idx].to(self.density_grid.device)
        else:
            density_grid = self.density_grid_list[0].to(self.density_grid.device)
        tmp_grid = - torch.ones_like(density_grid)

        # full update.
        if self.iter_density < 16:
        #if True:
            X = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Y = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)
            Z = torch.arange(self.grid_size, dtype=torch.int32, device=self.density_bitfield.device).split(S)

            for xs in X:
                for ys in Y:
                    for zs in Z:
                        
                        # construct points
                        xx, yy, zz = custom_meshgrid(xs, ys, zs)
                        coords = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1) # [N, 3], in [0, 128)
                        indices = raymarching.morton3D(coords).long() # [N]
                        xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]

                        # cascading
                        for cas in range(self.cascade):
                            bound = min(2 ** cas, self.bound)
                            half_grid_size = bound / self.grid_size
                            # scale to current cascade's resolution
                            cas_xyzs = xyzs * (bound - half_grid_size)
                            # add noise in [-hgs, hgs]
                            cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                            # query density
                            sigmas = self.density(cas_xyzs, max_weight=max_weight, choice=None if idx is None else idx - 1, bs_weight=bs_weight)['sigma'].reshape(-1).detach()
                            sigmas *= self.density_scale
                            # assign 
                            # print(tmp_grid[cas, indices].device)
                            # print(sigmas.device)
                            # print(self.density_grid[0].device)
                            tmp_grid[cas, indices] = sigmas.float()

        # partial update (half the computation)
        # TODO: why no need of maxpool ?
        else:
            N = self.grid_size ** 3 // 4 # H * H * H / 4
            for cas in range(self.cascade):
                # random sample some positions
                coords = torch.randint(0, self.grid_size, (N, 3), device=self.density_bitfield.device) # [N, 3], in [0, 128)
                indices = raymarching.morton3D(coords).long() # [N]
                # random sample occupied positions
                occ_indices = torch.nonzero(density_grid[cas] > 0).squeeze(-1) # [Nz]
                rand_mask = torch.randint(0, occ_indices.shape[0], [N], dtype=torch.long, device=self.density_bitfield.device)
                occ_indices = occ_indices[rand_mask] # [Nz] --> [N], allow for duplication
                occ_coords = raymarching.morton3D_invert(occ_indices) # [N, 3]
                # concat
                indices = torch.cat([indices, occ_indices], dim=0)
                coords = torch.cat([coords, occ_coords], dim=0)
                # same below
                xyzs = 2 * coords.float() / (self.grid_size - 1) - 1 # [N, 3] in [-1, 1]
                bound = min(2 ** cas, self.bound)
                half_grid_size = bound / self.grid_size
                # scale to current cascade's resolution
                cas_xyzs = xyzs * (bound - half_grid_size)
                # add noise in [-hgs, hgs]
                cas_xyzs += (torch.rand_like(cas_xyzs) * 2 - 1) * half_grid_size
                # query density
                sigmas = self.density(cas_xyzs, max_weight=max_weight, choice=None if idx is None else idx - 1, bs_weight=bs_weight)['sigma'].reshape(-1).detach()
                sigmas *= self.density_scale # 为1，大于1能improve sharp 
                # assign 
                tmp_grid[cas, indices] = sigmas.float()


        # ema update
        # density_grid = tmp_grid
        valid_mask = (density_grid >= 0) & (tmp_grid >= 0)
        density_grid[valid_mask] = torch.maximum(density_grid[valid_mask] * decay, tmp_grid[valid_mask])

        if idx is not None:
            self.density_grid_list[idx] = density_grid
        else:
            self.iter_density += 1
            self.density_grid_list[0] = density_grid

            ### update step counter
            total_step = min(16, self.local_step)
            if total_step > 0:
                self.mean_count = int(self.step_counter[:total_step, 0].sum().item() / total_step)
            self.local_step = 0

    def m_encoder(self, x:torch.tensor, max_weight=None, choice=None, bs_weight=None):
        if choice is None:
            if bs_weight is not None: # [B, 46]

                # H = torch.stack(self.H[1:47]) # [46, offset, level_dim]
                H = torch.stack([self.H[i] for i in range(1, 47)]) # [46, offset, level_dim]
                if bs_weight.shape[0] > 1:
                    e = self.H[0] + torch.einsum('ij,jkl->ikl', [bs_weight, H]).mean(0)
                else:
                    e = self.H[0] + torch.einsum('ij,jkl->ikl', [bs_weight, H]).squeeze(0) # test is [0]
                    
                x = self.encoder(x, embeddings=e)
            else:
                x = self.encoder(x, embeddings=self.H[0])
        else:
            x = self.encoder(x, embeddings=self.H[0] + max_weight[choice] * self.H[choice + 1])
        return x

    def forward(self, x, d, bs_weight=None):
        # x: [N, 3], in [-bound, bound]
        # d: [N, 3], nomalized in [-1, 1]

        # sigma
        x = self.m_encoder(x, bs_weight=bs_weight)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = F.softplus(h[..., 0])
        geo_feat = h[..., 1:]

        # color
        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        color = torch.sigmoid(h)


        return sigma, color

    def density(self, x, max_weight=None, choice=None, bs_weight=None):
        # x: [N, 3], in [-bound, bound]

        # x = self.encoder(x, bound=self.bound)
        x=x.to(dtype=torch.float32)
        x = self.m_encoder(x, max_weight=max_weight, choice=choice, bs_weight=bs_weight)

        h = x
        for l in range(self.num_layers):
            h = self.sigma_net[l](h)
            if l != self.num_layers - 1:
                h = F.relu(h, inplace=True)

        sigma = F.relu(h[..., 0])
        # sigma = trunc_exp(h[..., 0])
        geo_feat = h[..., 1:]

        return {
            'sigma': sigma,
            'geo_feat': geo_feat,
        }

    def background(self, x, d):
        # x: [N, 2], in [-1, 1]

        h = self.encoder_bg(x) # [N, C]
        d = self.encoder_dir(d)

        h = torch.cat([d, h], dim=-1)
        for l in range(self.num_layers_bg):
            h = self.bg_net[l](h)
            if l != self.num_layers_bg - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        rgbs = torch.sigmoid(h)

        return rgbs

    # allow masked inference
    def color(self, x, d, mask=None, geo_feat=None, **kwargs):
        # x: [N, 3] in [-bound, bound]
        # mask: [N,], bool, indicates where we actually needs to compute rgb.

        if mask is not None:
            rgbs = torch.zeros(mask.shape[0], 3, dtype=x.dtype, device=x.device) # [N, 3]
            # in case of empty mask
            if not mask.any():
                return rgbs
            x = x[mask]
            d = d[mask]
            geo_feat = geo_feat[mask]

        d = self.encoder_dir(d)
        h = torch.cat([d, geo_feat], dim=-1)
        for l in range(self.num_layers_color):
            h = self.color_net[l](h)
            if l != self.num_layers_color - 1:
                h = F.relu(h, inplace=True)
        
        # sigmoid activation for rgb
        h = torch.sigmoid(h)

        if mask is not None:
            rgbs[mask] = h.to(rgbs.dtype) # fp16 --> fp32
        else:
            rgbs = h

        return rgbs        

    # optimizer utils
    def get_params(self, lr):

        params = [
            {'params': self.encoder.parameters(), 'lr': lr},
            {'params': self.H.parameters(), 'lr': lr},
            {'params': self.sigma_net.parameters(), 'lr': lr},
            {'params': self.encoder_dir.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
        ]
        if self.bg_radius > 0:
            params.append({'params': self.encoder_bg.parameters(), 'lr': lr})
            params.append({'params': self.bg_net.parameters(), 'lr': lr})
        
        return params
