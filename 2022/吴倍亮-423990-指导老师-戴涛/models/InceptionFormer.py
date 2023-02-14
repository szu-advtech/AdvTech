import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.registry import register_model
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, Mlp


class PatchEmbedding(nn.Module):
    def __init__(self, kernel_size=3, stride=2, padding=0, in_channels=3, out_channels=96):
        """
        patch embedding, 卷积嵌入，缩小输入尺寸，Vit
        """
        super().__init__()
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        """
        :param X: B, C, H, W
        :return: B, H, W, C
        """
        X = self.proj(X)
        X = self.norm(X)
        return X.permute(0, 2, 3, 1)


class Left(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.left = nn.Sequential(  # 左侧高频处理，最大池化（spatial）、卷积（channel）
            nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=padding),
            nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=1, stride=1, padding=0),
            nn.GELU()
        )

    def forward(self, X):
        """
        :param X: B, C, H, W
        :return:B, C, H, W
        """
        return self.left(X)


class Mid(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.mid = nn.Sequential(  # 中间高频处理，linear（channel）、depth（spatial）
            nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=dim, out_channels=dim,
                      kernel_size=kernel_size, stride=stride, padding=padding,
                      bias=False, groups=dim),
            nn.GELU()
        )

    def forward(self, X):
        """
        :param X: B, C, H, W
        :return:B, C, H, W
        """
        return self.mid(X)


class Right(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., pool_size=2):
        """
        base on basic attention module from internet and have some changes.
        """
        super().__init__()
        self.AvgPool = nn.AvgPool2d(kernel_size=pool_size, stride=pool_size,
                                    padding=0) if pool_size > 1 else nn.Identity()

        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.upsample = nn.Upsample(scale_factor=pool_size) if pool_size > 1 else nn.Identity()

    def forward(self, X):
        """
        :param X: B, C, H, W
        :return: B, C, H, W
        """
        X = self.AvgPool(X)
        B, C, H, W = X.shape
        N = H * W
        X = torch.flatten(X, start_dim=2).transpose(-2, -1)  # (B, N, C)
        qkv = self.qkv(X).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        X = (attn @ v).transpose(2, 3).reshape(B, C, N).view(B, C, H, W)
        return self.upsample(X)


class InceptionMixer(nn.Module):
    def __init__(self, dim, num_parts=8, qkv_bias=False, attn_drop=0., proj_drop=0., attention_head=1, pool_size=2):
        super().__init__()
        self.num_heads = num_parts
        assert dim % num_parts == 0
        self.head_dim = dim // num_parts

        self.right_channels = attention_head * self.head_dim
        self.left_channels = self.mid_channels = (dim - self.right_channels) // 2

        self.left = Left(self.left_channels)
        self.mid = Mid(self.mid_channels)
        self.right = Right(self.right_channels, num_heads=attention_head,
                           qkv_bias=qkv_bias, attn_drop=attn_drop, pool_size=pool_size)

        self.conv_fuse = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False,
                                   groups=dim)
        self.proj = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, stride=1, padding=0)

        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)  # To (B,C,H,W)
        x_left = x[:, :self.left_channels, :, :]
        x_mid = x[:, self.left_channels:self.left_channels + self.mid_channels, :, :]
        x_right = x[:, self.left_channels + self.mid_channels:, :, :]

        x_left = self.left(x_left)
        x_mid = self.mid(x_mid)
        x_right = self.right(x_right)

        x = torch.cat((x_left, x_mid, x_right), dim=1)  # (B,C,H,W)
        x = x + self.conv_fuse(x)
        x = self.proj_drop(self.proj(x)).permute(0, 2, 3, 1)
        return x  # (B,H,W,C)


class IFormerBlock(nn.Module):
    """
    Implementation of one IFormer block.
    refer to https://github.com/sail-sg/poolformer
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, attention_head=1, pool_size=2,
                 token_mixer=InceptionMixer,
                 use_layer_scale=False, layer_scale_init_value=1e-5,
                 ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = token_mixer(dim=dim, num_parts=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
                                       attention_head=attention_head, pool_size=pool_size)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(mlp_ratio * dim), act_layer=nn.GELU, drop=drop)

        self.use_layer_scale = use_layer_scale
        if self.use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, X):
        if self.use_layer_scale:
            X = X + self.drop_path(self.layer_scale_1 * self.token_mixer(self.norm1(X)))
            X = X + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(X)))
        else:
            X = X + self.drop_path(self.token_mixer(self.norm1(X)))
            X = X + self.drop_path(self.mlp(self.norm2(X)))
        return X


def _get_pos_embed(pos_embed, num_patches_def, H, W):
    """
    add in each stage, not only in first time input.
    according to source code, change to trainable position embedding.
    """
    if H * W == num_patches_def * num_patches_def:
        return pos_embed
    else:
        return F.interpolate(
            pos_embed.permute(0, 3, 1, 2),
            size=(H, W), mode="bilinear", align_corners=True).permute(0, 2, 3, 1)


class Inception(nn.Module):
    """
    Implementation of Inception for token-mixer module
    refer to https://github.com/sail-sg/poolformer
    """

    def __init__(self, image_size=224, in_channels=3, num_classes=1000,
                 embedding_list=None, num_block=None, num_part=None,
                 mlp_ratio=4, qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, num_low_part=None,
                 use_layer_scale=False, layer_scale_init_value=1e-5
                 ):
        super().__init__()
        # block nums: 3 3 9 3
        # state index: 0 3 6 15 18
        st_idx = [0] + [sum(num_block[:i + 1]) for i in range(4)]
        pool_size = [2, 2, 1, 1]

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, st_idx[-1])]

        blocks = []  # blocks1、blocks2、blocks3、blocks4
        patchs_size = []  # 56 28 14 7
        pos_embeds = []  # 56 28 14 7
        for i in range(4):
            block = nn.Sequential(*[
                IFormerBlock(
                    dim=embedding_list[i], num_heads=num_part[i], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[j], norm_layer=norm_layer,
                    attention_head=num_low_part[j], pool_size=pool_size[i],
                    use_layer_scale=use_layer_scale, layer_scale_init_value=layer_scale_init_value
                )
                for j in range(st_idx[i], st_idx[i + 1])])
            patch_size = image_size // (4 * (i + 1))
            blocks.append(block)
            patchs_size.append(patch_size)
            pos_embeds.append(nn.Parameter(torch.zeros(1, patch_size, patch_size, embedding_list[i])))
        self.embed_first = PatchEmbedding(in_channels=in_channels, out_channels=embedding_list[0] // 2, padding=1)
        patch_embed = [PatchEmbedding(in_channels=embedding_list[0] // 2, out_channels=embedding_list[0], padding=1)] + \
                      [PatchEmbedding(kernel_size=2, in_channels=embedding_list[i], out_channels=embedding_list[i + 1])
                       for i in range(3)]

        self.blocks = nn.ModuleList(blocks)
        self.patch_embed = nn.ModuleList(patch_embed)
        self.pos_embeds = nn.ParameterList(pos_embeds)
        self.patchs_size = patchs_size

        self.num_classes = num_classes
        self.norm = norm_layer(embedding_list[3])
        self.head = nn.Linear(embedding_list[3], num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, X):
        B, C, H, W = X.shape
        X = self.embed_first(X)  # [B, H, W, C]
        for i in range(4):
            X = X.permute(0, 3, 1, 2)  # [B, C, H, W]
            X = self.patch_embed[i](X)
            B, H, W, C = X.shape
            X = X + _get_pos_embed(self.pos_embeds[i], self.patchs_size[i], H, W)
            X = self.blocks[i](X)
        X = X.flatten(1, 2)  # B, H, W, C To B, N, C
        X = self.norm(X)
        X = self.head(X.mean(1))  # B, C

        return X


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


@register_model
def iFormer_small(pretrained=False, **kwargs):
    num_parts = [3, 6, 10, 12]
    num_blocks = [3, 3, 9, 3]
    out_channels_list = [96, 192, 320, 384]
    num_low_part = [1] * 3 + [3] * 3 + [7] * 4 + [9] * 5 + [11] * 3
    model = Inception(image_size=224,
                      num_block=num_blocks,
                      embedding_list=out_channels_list,
                      num_part=num_parts,
                      num_low_part=num_low_part,
                      use_layer_scale=True, layer_scale_init_value=1e-6,
                      )
    model.default_cfg = _cfg(crop_pct=0.9)
    if pretrained:
        # url = model_urls['iFormer_test']
        # checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cpu", check_hash=True)
        checkpoint = torch.load("/mnt/nas/wbl/poolformer-main/output/train/20230208-103247-iFormer_small-224/last.pth.tar")
        model.load_state_dict(checkpoint['state_dict'])
    return model
