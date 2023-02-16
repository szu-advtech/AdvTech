import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional

def window_partition(x, window_size: int):
    """
    将feature map按照window_size划分成一个个没有重叠的window
    Args:
        x: (B, H, W, C)
        window_size (int): window size(M)

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    # permute: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H//Mh, W//Mh, Mw, Mw, C] 重新排列
    # view: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B*num_windows, Mh, Mw, C]
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)  # contiguous使数据变连续
    return windows


def window_reverse(windows, window_size: int, H: int, W: int):
    """
    将一个个window还原成一个feature map；上面的一个逆过程
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size(M)
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))  # 得到batch数
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbed(nn.Module):
    """
    对应着网络架构中 patch partition + linear embedding

    由于我们不能将像素作为Transformer的输入，所以vit将其划分为196个16*16*3大小的patch（flatten为768），然后输入token序列[196,768]
    而swin-T中不同，不以16*16为patch切割大小，而是以4*4为patch大小，然后通过后续的patch merging不断增大每个patch的大小
    """
    def __init__(self, patch_size=4, in_c=3, embed_dim=96, norm_layer=None):
        # ViT中patch_size=16*16，ST中patch_size=4*4

        super().__init__()
        patch_size = (patch_size, patch_size)  # 4*4
        self.patch_size = patch_size
        self.in_chans = in_c
        self.embed_dim = embed_dim

        # 原文操作：将[224,224,3]的图片，划分为56*56=3136个patch，shape=[4,4,3]，然后本应flatten为48，此处flatten为96(一个token的长度)，故而得到[56*56,96]
        # 具体实现：直观上可以直接将image按4x4大小的patch直接切开，分成56*56个patch，但是实际实现是使用4*4的卷积得到56*56个patch
        # 卷积核[96,4,4,stride=4]（直接一步到位96），卷积层输入为[3,224,224](实际上默认图片按CHW排序),输出为[96,56,56]，然后转置后flatten为[56*56,96] 即[num_tokens, token]

        self.proj = nn.Conv2d(in_channels=in_c, out_channels=embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()  # nn.Identity()函数建立一个输入模块，啥都不做，恒等映射

    def forward(self, x):
        _, _, H, W = x.shape  # 获取图片的高宽；batch, channel, height, width

        # padding
        # 如果输入图片的H，W不是patch_size的整数倍，需要进行padding
        pad_input = (H % self.patch_size[0] != 0) or (W % self.patch_size[1] != 0)
        if pad_input:
            # 此时的x [B, C, H, W]
            # (W_left, W_right, H_top, H_bottom, C_front, C_back)
            # 左边填0列，右边填充4-W%4列，上边填充0行，下边填充4-H%4行，前面和后面都不填充
            x = F.pad(input=x,
                      pad=(0, self.patch_size[1] - W % self.patch_size[1], 0, self.patch_size[0] - H % self.patch_size[0], 0, 0))

        # 下采样patch_size倍:对于一个样值序列间隔4个样值取样一次，降采样后整体图片的高度和宽度缩小了四倍
        # proj: [B, C, H, W] (B, 3, 224, 224) -> [B, C, H, W] (B, 96, 56, 56)
        # 一定注意此处的56并非一个patch，而是深度方向上是一个patch
        x = self.proj(x)  # proj是一个可调用对象
        _, _, H, W = x.shape  # 记录下采样之后整体图片的的高度和宽度
        # flatten: [B, C, H, W] (B, 96, 56, 56) -> [B, C, HW] (B, 96, 3136)
        # transpose: [B, C, HW] (B, 96, 3136)-> [B, HW, C] (B, 3136, 96) [num_tokens. one_token_length]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchMerging(nn.Module):
    """ Patch Merging Layer.

    Args:
        dim (int): 输入数据的通道维度，为什么要叫dim，有点怪
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.norm = norm_layer(4 * dim)  # 在linear层之前使用
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)  # 根据图片可知，最后一步Linear时输入channel是原始的4倍；图：https://github.com/haoxia1/DeepLearning-Notes/blob/main/images/d0c3e817509740de68877f204caca1d4c02e59d290cb8d2355e5129c4773ca16.png

    def forward(self, x, H, W):
        """
        x: B, H*W, C 这里的x是不知道高宽值的
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # padding
        # 因为patchMerging需要下采样两倍，如果输入feature map的H，W不是2的整数倍，需要进行padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            # 注意这里的Tensor通道是[B, H, W, C]，所以会和官方文档有些不同
            # 注意pad函数是pad最后三个维度，且从后往前设置，此处后三个维度HWC，从后往前设置CWH
            # (C_front, C_back, W_left, W_right, H_top, H_bottom)
            # 若非偶数，在右侧或下侧padding一行0
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        # 以2x2大小为一个窗口，将窗口中相同位置上的像素取出，得到四个矩阵（这里有趣）
        x0 = x[:, 0::2, 0::2, :]  # [B, H/2, W/2, C] 蓝色
        x1 = x[:, 1::2, 0::2, :]  # [B, H/2, W/2, C] 绿色
        x2 = x[:, 0::2, 1::2, :]  # [B, H/2, W/2, C] 黄色
        x3 = x[:, 1::2, 1::2, :]  # [B, H/2, W/2, C] 红色
        x = torch.cat([x0, x1, x2, x3], -1)  # [B, H/2, W/2, 4*C] # 按channel维度拼接起来
        x = x.view(B, -1, 4 * C)  # [B, H/2*W/2, 4*C] 将HW维度展平，将拼接后的特征图恢复到输入时的形式[B, H*W, C]

        x = self.norm(x)
        x = self.reduction(x)  # [B, H/2*W/2, 2*C] Linear层

        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    详见ViT笔记
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # [Mh, Mw]
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # [2*Mh-1 * 2*Mw-1, nH]

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # [2, Mh, Mw]
        coords_flatten = torch.flatten(coords, 1)  # [2, Mh*Mw]
        # [2, Mh*Mw, 1] - [2, 1, Mh*Mw]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # [2, Mh*Mw, Mh*Mw]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [Mh*Mw, Mh*Mw, 2]
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # [Mh*Mw, Mh*Mw]
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: input features with shape of (num_windows*B, Mh*Mw, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        # [batch_size*num_windows, Mh*Mw, total_embed_dim]
        B_, N, C = x.shape
        # qkv(): -> [batch_size*num_windows, Mh*Mw, 3 * total_embed_dim]
        # reshape: -> [batch_size*num_windows, Mh*Mw, 3, num_heads, embed_dim_per_head]
        # permute: -> [3, batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        # transpose: -> [batch_size*num_windows, num_heads, embed_dim_per_head, Mh*Mw]
        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, Mh*Mw]
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # relative_position_bias_table.view: [Mh*Mw*Mh*Mw,nH] -> [Mh*Mw,Mh*Mw,nH]
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # [nH, Mh*Mw, Mh*Mw]
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            # mask: [nW, Mh*Mw, Mh*Mw]
            nW = mask.shape[0]  # num_windows
            # attn.view: [batch_size, num_windows, num_heads, Mh*Mw, Mh*Mw]
            # mask.unsqueeze: [1, nW, 1, Mh*Mw, Mh*Mw]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # @: multiply -> [batch_size*num_windows, num_heads, Mh*Mw, embed_dim_per_head]
        # transpose: -> [batch_size*num_windows, Mh*Mw, num_heads, embed_dim_per_head]
        # reshape: -> [batch_size*num_windows, Mh*Mw, total_embed_dim]
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    每个stage中的SwinTransformerBlock
    """
    def __init__(self, dim, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"  # 一般是1/2向下取整

        self.norm1 = norm_layer(dim)  # 1. 论文中的LN
        # 2. 这个windowAttention是W-MSA或者SW-MSA；shift_size=0为W-MSA，大于0为SW-MSA
        self.attn = WindowAttention(
            dim, window_size=(self.window_size, self.window_size), num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()  # 3. droup层
        self.norm2 = norm_layer(dim)  # 第二个LN
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, attn_mask):
        H, W = self.H, self.W
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # pad feature maps to multiples of window size
        # 把feature map给pad到window size的整数倍
        pad_l = pad_t = 0
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # [nW*B, Mh, Mw, C]
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # [nW*B, Mh*Mw, C]

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=attn_mask)  # [nW*B, Mh*Mw, C]

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)  # [nW*B, Mh, Mw, C]
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # [B, H', W', C]

        # reverse cyclic shift
        if self.shift_size > 0:  # 即SW-MSA
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            # 把前面pad的数据移除掉
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class BasicLayer(nn.Module):
    """
    A basic Swin Transformer layer for one stage.
    应该是一个stage，里面有多个block + 一个patchMerging
    """

    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size
        self.use_checkpoint = use_checkpoint
        self.shift_size = window_size // 2  # 使用SW-MSA的时候向右向下偏移多少个像素；窗口除以2再向下取整；

        # build blocks 这个block是当前stage的STBlock数
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else self.shift_size,  # 比如第一个stage有2个block，i=0时，使用W-MSA，i=1时使用SW-MSA
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer) for i in range(depth)])

        # patch merging layer
        if downsample is not None:  # downsample是PatchMerging类的实例
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def create_mask(self, x, H, W):
        # calculate attention mask for SW-MSA
        # 保证Hp和Wp是window_size的整数倍 （向上取整）
        Hp = int(np.ceil(H / self.window_size)) * self.window_size
        Wp = int(np.ceil(W / self.window_size)) * self.window_size
        # 拥有和feature map一样的通道排列顺序，方便后续window_partition
        img_mask = torch.zeros((1, Hp, Wp, 1), device=x.device)  # [1, Hp, Wp, 1]  batch数是1

        # 下面参考图片：https://github.com/haoxia1/DeepLearning-Notes/blob/main/images/18e9dc14c1ceced0c0fdeb66e6f309bbe79c825b6da386739952a240c401a034.png
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))  # 高度方向：分为三块，[0,1,2]（7行）,[3,4,5]（4行）,[6,7,8]（3行）
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))  # 宽度方向：分为三块，[0,3,6]（7列）[1,4,7]（4列）,[2,5,8]（3列）
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                # 遍历的时候，首先得到的是第0块，数值对应的设置为0，然后是右边的第1块，数值对应的设置为1
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # 现在我们得到了0,1,..,9的划分，然后

        mask_windows = window_partition(img_mask, self.window_size)  # [nW, Mh, Mw, 1]  # 将image按照7*7划分成一个个窗口
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)  # [nW, Mh*Mw]
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # [nW, 1, Mh*Mw] - [nW, Mh*Mw, 1] # unsqueeze增维 增加大小为1的维度，对输入的指定位置插入维度为1
        # 上一行解释：https://www.bilibili.com/video/BV1yg411K7Yc?t=2346.0  掩码的实现
        # [nW, Mh*Mw, Mh*Mw]
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))  # 对于不等于0的区域填入负100
        return attn_mask

    def forward(self, x, H, W):
        attn_mask = self.create_mask(x, H, W)  # [nW, Mh*Mw, Mh*Mw]  # 使用SW-MSA时需要用到mask
        for blk in self.blocks:
            blk.H, blk.W = H, W
            if not torch.jit.is_scripting() and self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, attn_mask)
            else:  # 默认跳到这
                x = blk(x, attn_mask)
        if self.downsample is not None:
            x = self.downsample(x, H, W)
            H, W = (H + 1) // 2, (W + 1) // 2  # 下采样2倍后的图片

        return x, H, W


class SwinTransformer(nn.Module):
    r""" Swin Transformer

    Args:
        patch_size (int | tuple(int)): 下采样倍数，默认高度和宽度都下采样四倍
        in_chans (int): 输入图片通道数. Default: 3
        num_classes (int): 分类类别(分类头)数. Default: 1000
        embed_dim (int): Linear Embedding后得到的通道数 C
        depths (tuple(int)): 每一个stage中重复Swin-Transformer block的次数；默认配置为Swin-T
        num_heads (tuple(int)): ST中multi-head self-attention的head的个数
        window_size (int): MSA采用的窗口大小
        mlp_ratio (float): MLP中第一个全连接层将channel翻多少倍
        qkv_bias (bool): MSA中是否使用偏置  Default: True
        drop_rate (float): 在PatchEmbed层之后
        attn_drop_rate (float): MSA中使用；Attention dropout rate. Default: 0
        drop_path_rate (float): 在每一个ST中所采用的drop rate；从0逐渐递增的. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to **save memory**. Default: False
    """

    def __init__(self, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24),
                 window_size=7, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False, **kwargs):

        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)  # 对应stage个数
        self.embed_dim = embed_dim  # C
        self.patch_norm = patch_norm
        # stage4输出特征矩阵的channels
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))  # 8倍
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches 对应着patch partition + linear embedding
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)  # 实例化一个patchEmbed层，需给定init的参数
        self.pos_drop = nn.Dropout(p=drop_rate)

        # swinTransformerblock中的drop_path_rate是从0慢慢增长到指定的值
        dpr = [x.item() for x in torch.linspace(start=0, end=drop_path_rate, steps=sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()  # 将所有层放在ModuleList当中
        for i_layer in range(self.num_layers):  # 4
            # 注意这里构建的stage和论文图中有些差异
            # 这里的stage不包含该stage的patch_merging层，包含的是下个stage的patch_embedding层；因为第一个linear embedding被放到patch embedding中了嘛
            layers = BasicLayer(dim=int(embed_dim * 2 ** i_layer),  # 输入的dim是逐渐乘2的，first stage：96*(2^0); 2 stage:96*(2^1)
                                depth=depths[i_layer],  # 这个stage中要堆叠多少次STblock
                                num_heads=num_heads[i_layer],  # 同上
                                window_size=window_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                                norm_layer=norm_layer,
                                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,  # 是下一个stage中的patchMerging（此处判断表示第四个stage后面是没有了的）
                                use_checkpoint=use_checkpoint)
            self.layers.append(layers)

        # 对于分类而言，我们还需要加一个LayerNorm层和一个自适应的全局平均池化
        self.norm = norm_layer(self.num_features)  # 输入是num_features 即stage4的输出的channel
        self.avgpool = nn.AdaptiveAvgPool1d(1)  # 池化后高宽变为1 ？？？
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()  # 全连接层[B,768]->[B,1000]

        self.apply(self._init_weights)  # 运用apply调用下面的方法对模型进行权重初始化

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 输入的x是[B, C, H, W]
        # patch_embed后: [B, L, C] = [B, H * W, C]
        x, H, W = self.patch_embed(x) # 下采样四倍，通道数变为96；实例patch_embed是一个可调用对象，其调用了类中的forward方法
        # 按一定的比例随机丢失
        x = self.pos_drop(x)

        for layer in self.layers:  # 依次通过stage1234
            x, H, W = layer(x, H, W)

        x = self.norm(x)  # [B, L, C] stage4之后跟一个LN
        x = self.avgpool(x.transpose(1, 2))  # [B, C, 1]
        x = torch.flatten(x, 1) # flatten将1维及其之后的都flatten 得到[B, C]
        x = self.head(x)
        return x


def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

def swin_tiny_patch4_window7_224(num_classes: int = 1000, **kwargs):
    # 在ImageNet-1K上的预训练权重
    # https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            **kwargs)
    return model