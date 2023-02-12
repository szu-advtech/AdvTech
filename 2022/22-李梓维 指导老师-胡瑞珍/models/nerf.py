import torch
from torch import nn

import tinycudann as tcnn
import vren
from einops import rearrange
#from .custom_functions import TruncExp
import numpy as np

#from .rendering import NEAR_DISTANCE
from torch.cuda.amp import custom_fwd, custom_bwd
from torch_scatter import segment_csr
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

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (x, sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, typ,
                 D=2, W=64, skips=[8],
                 in_channels_xyz=3, in_channels_dir=3,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 beta_min=0.03):

        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = 3
        self.in_channels_dir = 3

        self.hash_xyz = 16
        self.hash_dir = 16

        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = False if typ=='coarse' else encode_transient
        self.in_channels_t = in_channels_t
        self.beta_min = beta_min



        ##NGP
        self.rgb_act = 'Sigmoid'

        # scene bounding box
        self.scale = 0.5
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*self.scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*self.scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*self.scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*self.scale/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        #NGP

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
                encoding_config={
                    "otype": "Grid",
                    "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": self.rgb_act,
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )

        if self.rgb_act == 'None':  # rgb_net output is log-radiance
            for i in range(3):  # independent tonemappers for r,g,b
                tonemapper_net = \
                    tcnn.Network(
                        n_input_dims=1, n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "Sigmoid",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        }
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)


        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            # elif i in skips:
            #     layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"xyz_encoding_{i+1}", layer)
        self.xyz_encoding_final = nn.Linear(self.hash_xyz, self.hash_xyz)

        # direction encoding layers
        self.dir_encoding = nn.Sequential(
                        nn.Linear(self.hash_xyz+self.hash_dir+self.in_channels_a, 64), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Sequential(nn.Linear(self.hash_xyz, 1), nn.Softplus())
        self.static_rgb = nn.Sequential(nn.Linear(64, 64), nn.Sigmoid(),
                                        nn.Linear(64, 3), nn.Sigmoid())

        if self.encode_transient:
            # transient encoding layers
            self.transient_encoding = nn.Sequential(
                                        nn.Linear(self.hash_xyz+in_channels_t, 64), nn.ReLU(True),
                                        nn.Linear(64, 64), nn.ReLU(True),
                                        #nn.Linear(W//2, W//2), nn.ReLU(True),
                                        nn.Linear(64, 64), nn.ReLU(True))
            # transient output layers
            self.transient_sigma = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
            self.transient_rgb = nn.Sequential(nn.Linear(64, 3), nn.Sigmoid())
            self.transient_beta = nn.Sequential(nn.Linear(64, 1), nn.Softplus())

    def density(self, x, return_feat=False):
        x = (x - self.xyz_min) / (self.xyz_max - self.xyz_min)
        h = self.xyz_encoder(x)
        sigmas = TruncExp.apply(h[0])
        if return_feat: return sigmas, h
        return sigmas

    def forward(self, x, sigma_only=False, output_transient=True):

        if sigma_only:
            input_xyz = x
        elif output_transient:
            input_xyz, input_dir,input_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir,self.in_channels_a,
                                self.in_channels_t], dim=-1)
        else:
            # print(x.shape)
            # print(self.in_channels_xyz)
            # print(self.in_channels_dir)
            # print(self.in_channels_a)
            input_xyz, input_dir,input_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir,self.in_channels_a], dim=-1)

        input_dir = input_dir/torch.norm(input_dir, dim=1, keepdim=True)
        input_dir = self.dir_encoder((input_dir+1)/2)
        input_dir_a=torch.cat([input_dir, input_a], 1)
        static_sigma, input_xyz = self.density(input_xyz, return_feat=True)
        # input_xyz = self.density(input_xyz, return_feat=True)
        xyz_ = input_xyz
        # for i in range(self.D):
        #     if i in self.skips:
        #         xyz_ = torch.cat([input_xyz, xyz_], 1)
        #     xyz_ = getattr(self, f"xyz_encoding_{i+1}")(xyz_)
        xyz_=xyz_.float()
        static_sigma = self.static_sigma(xyz_) # (B, 1)
        if sigma_only:
            return static_sigma
        #xyz_encoding_final = self.xyz_encoding_final(xyz_)
        xyz_encoding_final=xyz_
        dir_encoding_input = torch.cat([xyz_encoding_final, input_dir_a], 1)
        dir_encoding = self.dir_encoding(dir_encoding_input)
        static_rgb = self.static_rgb(dir_encoding) # (B, 3)
        static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

        if not output_transient:
            return static

        transient_encoding_input = torch.cat([xyz_encoding_final, input_t], 1)
        transient_encoding = self.transient_encoding(transient_encoding_input)
        transient_sigma = self.transient_sigma(transient_encoding) # (B, 1)
        transient_rgb = self.transient_rgb(transient_encoding) # (B, 3)
        transient_beta = self.transient_beta(transient_encoding) # (B, 1)

        transient = torch.cat([transient_rgb, transient_sigma,
                               transient_beta], 1) # (B, 5)

        return torch.cat([static, transient], 1) # (B, 9)