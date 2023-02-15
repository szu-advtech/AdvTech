import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from timm.models.layers import trunc_normal_
from einops import rearrange, repeat
from functools import partial

from .vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model

__all__ = [
    'deit_tscam_tiny_patch16_224', 'deit_tscam_small_patch16_224', 'deit_tscam_base_patch16_224',
]

def norm_cam(cam):
    # cam [B N]
    if len(cam.shape) == 3:
        cam = cam - repeat(rearrange(cam, 'B H W -> B (H W)').min(1,
                           keepdim=True)[0], 'B 1 -> B 1 1')
        cam = cam / repeat(rearrange(cam, 'B H W -> B (H W)').max(1,
                           keepdim=True)[0], 'B 1 -> B 1 1')
    elif len(cam.shape) == 2:
        cam = cam - cam.min(1, keepdim=True)[0]
        cam = cam / cam.max(1, keepdim=True)[0]
    elif len(cam.shape) == 4:
        B, C, H, W = cam.shape
        cam = rearrange(cam, 'B C H W -> (B C) (H W)')
        cam -= cam.min(1, keepdim=True)[0]
        cam /= cam.max(1, keepdim=True)[0]
        cam = rearrange(cam, '(B C) (H W) -> B C H W', B = B, H=H)
    return cam

class TSCAM(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.head = nn.Conv2d(self.embed_dim, self.num_classes, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.head.apply(self._init_weights)

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to return patch embedding outputs
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, 1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn_weights = []
        for blk in self.blocks:
            x, weights = blk(x)
            attn_weights.append(weights)

        x = self.norm(x)
        return x[:, 0], x[:, 1:], attn_weights

    def forward(self, x, return_cam=False):
        x_cls, x_patch, attn_weights = self.forward_features(x)
        n, p, c = x_patch.shape
        x_patch = torch.reshape(x_patch, [n, int(p**0.5), int(p**0.5), c])
        x_patch = x_patch.permute([0, 3, 1, 2])
        x_patch = x_patch.contiguous()
        x_patch = self.head(x_patch)
        x_logits = self.avgpool(x_patch).squeeze(3).squeeze(2)
    
        if self.training:
           return x_logits
        else:
            attn_weights = torch.stack(attn_weights)        # 12 * B * H * N * N
            attn_weights = torch.mean(attn_weights, dim=2)  # 12 * B * N * N

            feature_map = x_patch.detach().clone()    # B * C * 14 * 14
            n, c, h, w = feature_map.shape
            cams = attn_weights.sum(0)[:, 0, 1:].unsqueeze(1).reshape(n, 1, h, w)
            cams = cams.reshape(n, 1, h*w)
            # feature_conv = torch.matmul(attn_maps.unsqueeze(1), feature_conv.reshape(n, 200, h*w, 1)).reshape(n,200,h,w)
            attn_maps = attn_weights.sum(0)[:, 1:, 1:] # B n n
            aggr_cams = torch.matmul(cams, attn_maps).reshape(n, 1, h, w)
            attn_cams = torch.matmul(attn_maps.unsqueeze(1), cams.unsqueeze(-1)).reshape(n, 1, h, w)
            out = aggr_cams + attn_cams
           
            # cams = attn_maps.mean(1, keepdim=True).reshape(n, 1, h, w)
            out = norm_cam(out)
            out = out * feature_map                           # B * C * 14 * 14

            return x_logits, out

@register_model
def deit_tscam_tiny_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()

        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

@register_model
def deit_tscam_small_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
    return model


@register_model
def deit_tscam_base_patch16_224(pretrained=False, **kwargs):
    model = TSCAM(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
            map_location="cpu", check_hash=True
        )['model']
        model_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint and checkpoint[k].shape != model_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint[k]

        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

