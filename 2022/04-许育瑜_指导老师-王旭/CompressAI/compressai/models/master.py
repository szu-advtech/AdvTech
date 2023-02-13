import imp
import torch
import torch.nn as nn
import math
import warnings
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_  # timm库，已经封装了部分transformer的相关的工具
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck,GaussianConditional
from compressai.entropy_models.entropy_models import EntropyModel
from compressai.layers import GDN, MaskedConv2d
from compressai.models import MeanScaleHyperprior,CompressionModel

from torchinfo import summary



# ==============================================feature encoder/decoder==========================================
# ==============================================feature encoder/decoder==========================================

def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """1x1 convolution."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride)

def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    """3x3 convolution with padding."""
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)

class ResidualBlock(nn.Module):
    """Simple residual block with two 3x3 convolutions.

    Args:
        in_ch (int): number of input channels
        out_ch (int): number of output channels
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.leaky_relu = nn.LeakyReLU(inplace=True)  # leakyReLU是在负数阶段，有一点点偏移于0
        self.conv2 = conv3x3(out_ch, out_ch)
        # 如果输入和输出的维度不一样，才需要进行1x1卷积改变通道
        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.leaky_relu(out)
        out = self.conv2(out)
        out = self.leaky_relu(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out


# input:
# depth_image: W,H ==>stride=1
# rgb_imgae: 2W,2H ==>stride=2
# in_channel:3
# out_channel:64
class Feature_encoder(nn.Module):
    def __init__(self,in_channel=3,out_channel=64,stride=1) -> None:
        super().__init__()
        self.conv1 = conv3x3(in_channel,out_channel,stride)
        self.resblock1 = ResidualBlock(64,64)
        self.resblock2 = ResidualBlock(64,64)
        self.resblock3 = ResidualBlock(64,64)

    def forward(self,x):
        out = self.conv1(x)
        shortcut = out
        out = self.resblock1(out)
        out = self.resblock2(out)
        out = self.resblock3(out)

        out = out + shortcut
        return out

# 反卷积也能够改变通道，但主要是为了恢复卷积之前的图片尺寸
def deconv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,  # 这是具有针对性的padding
        padding=kernel_size // 2,
    )

# in_channel:64
# out_channel:3
class Feature_decoder(nn.Module):
    def __init__(self,in_channel=64*3,out_channel=3,stride=1) -> None:
        super().__init__()
        
        self.resblock1 = ResidualBlock(in_channel,64)
        self.resblock2 = ResidualBlock(64,64)
        self.resblock3 = ResidualBlock(64,64)
        self.deconv1 = deconv(64,out_channel,kernel_size=3,stride=stride)
        self.conv = conv1x1(in_ch=in_channel,out_ch=64)

    def forward(self,x):
        shortcut = x

        out = self.resblock1(x)
        out = self.resblock2(out)
        out = self.resblock3(out)
        out = out + self.conv(shortcut)
        out = self.deconv1(out)
        
        return out

# ==============================================feature encoder/decoder==========================================


# ==============================================channle-wise alignment==========================================
# ==============================================channle-wise alignment==========================================

# class Channel_aligner(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.conv1=conv3x3(64,256)
#         self.leaky_relu = nn.LeakyReLU(inplace=True)
#         self.conv2=conv3x3(256,64)
#         self.conv3=conv3x3(256,64)
#         self.avgpool1=nn.AdaptiveAvgPool2d(1)  # adaptive的参数是输出的size，而普通的参数则是过程的kernel_size
#         self.avgpool2=nn.AdaptiveAvgPool2d(1)


#     # feature2 is guided 
#     def forward(self,feature1,feature2):
#         identity = feature2

#         out1 = self.conv1(feature1)
#         out1 = self.leaky_relu(out1)
#         out1 = self.conv2(out1)
#         beta = self.avgpool1(out1)
        
#         out2 = self.conv1(feature2)
#         out2 = self.leaky_relu(out2)
#         out2 = self.conv3(out2)
#         gamma = self.avgpool2(out2)

#         # 池化==》广播
#         out = gamma * identity + beta # 这里是否直接通过乘法和加法来实现？ # 这里通过广播来实现？
#         # print("beta,gamma:")
#         # print(beta,gamma)
#         return out,beta,gamma         # beta,gamma是要直接进行传输的【这里每个不同的样本对应不同的beta，gamma？这个在解码的时候应该如何使用？】


class Channel_aligner(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1=conv3x3(64,256)
        self.leaky_relu1 = nn.LeakyReLU(inplace=True)
        self.conv2=conv3x3(256,256)
        self.leaky_relu2 = nn.LeakyReLU(inplace=True)
        self.conv3=conv3x3(256,256)
        self.leaky_relu3 = nn.LeakyReLU(inplace=True)  
        self.conv4=conv3x3(256,256)
        self.leaky_relu4 = nn.LeakyReLU(inplace=True)



        self.conv5=conv3x3(256,64)
        self.conv6=conv3x3(256,64)
        self.avgpool1=nn.AdaptiveAvgPool2d(1)  # adaptive的参数是输出的size，而普通的参数则是过程的kernel_size
        self.avgpool2=nn.AdaptiveAvgPool2d(1)


    # feature2 is guided 
    def forward(self,feature1,feature2):
        identity = feature2

        out1 = self.conv1(feature1)
        out1 = self.leaky_relu1(out1)
        out2 = self.conv2(out1)
        out2 = self.leaky_relu2(out2)
        out3 = self.conv3(out2)
        out3 = self.leaky_relu3(out3)
        out4 = self.conv4(out3)
        out4 = self.leaky_relu4(out4)

        out5 = self.conv5(out4)
        beta = self.avgpool1(out5)
        
        out6 = self.conv1(feature2)
        out6 = self.leaky_relu1(out6)
        out7 = self.conv2(out6)
        out7 = self.leaky_relu2(out7)
        out8 = self.conv3(out7)
        out8 = self.leaky_relu3(out8)
        out9 = self.conv4(out8)
        out9 = self.leaky_relu4(out9)

        out10 = self.conv6(out9)
        gamma = self.avgpool2(out10)

        # 池化==》广播
        out = gamma * identity + beta # 这里是否直接通过乘法和加法来实现？ # 这里通过广播来实现？
        # print("beta,gamma:")
        # print(beta,gamma)
        return out,beta,gamma         # beta,gamma是要直接进行传输的【这里每个不同的样本对应不同的beta，gamma？这个在解码的时候应该如何使用？】

# ==============================================channle-wise alignment==========================================


# ==============================================Encoder==========================================
# ==============================================Encoder==========================================
def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

# input: [align_feature,x_feature]
class Master_encoder(CompressionModel): 
    def __init__(self,N=192,M=192,channel=64*2) -> None:
        super().__init__(M)
    
        self.g_a = nn.Sequential(
            conv(channel, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),  # in,out,kernel_size
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),  # 2M
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)


    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)  
        gaussian_params = self.entropy_parameters(   
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        # 因为master-decoder比较特别需要重写
        # x_hat = self.g_s(y_hat)

        return {
            "y_hat": y_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # 实际上这里可以进行优化，压缩的时候，是可以在GPU上进行的
    def compress(self, x):

        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)            # h_s,还需要和context，以及entropy_parameter共用才行

        s = 4                               # scaling factor between z and y
        kernel_size = 5                     # context prediction kernel size
        padding = (kernel_size - 1) // 2    # padding = 2   

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):          # 批次中的每一个
            string = self._compress_ar(
                y_hat[i: i + 1],            # 有冒号，只是为了保证维度
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        masked_weight = self.context_prediction.weight * self.context_prediction.mask   # context实际上由一个mask卷积核组成，而全通道共享一个卷积核
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]          
                ctx_p = F.conv2d(                                                       # 一个mask_convolution卷积的结果就是一个数值
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]                                    
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)          

                y_crop = y_crop[:, :, padding, padding]                                 # 取出所需要编码的像素，其他部分都只是padding
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)  
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())                             # extend会将添加项，修改成可迭代类型；append则是直接原封不动地添加
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

            # 更新scale_table和gs.update
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()  # 0.11~256，64格
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)  # 更新真实的高斯熵编码模型
        updated |= super().update(force=force)                                            # 更新entropy_bottleneck模型
        return updated

# ==============================================Encoder==========================================


# ==============================================Decoder==========================================
# ==============================================Decoder==========================================
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int, int): Image size.  Default: 224,224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=(224,224), patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        # 本来这里可以进行归一化层，但是rgbt论文中，是输出embeding的结果，然后再进行layernorm的
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C  # flatten(2):2,...,-1 维度进行展平
        if self.norm is not None:
            x = self.norm(x)
        return x

    # 计算复杂度
    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops



def window_partition(x, window_size=4):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class Mlp(nn.Module):
    # in_feature?hidden_feature,out_feature
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 相当于矩阵乘法：[AxB]x[BxC]==>[AxC],输入输出的维度为：BLC 【并不像2D的BCHW】
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5  # or，并不是执行逻辑与操作，而是选择一个非None，非0，非False的数值

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv1 = nn.Linear(dim, dim , bias=qkv_bias)     # 线性变换，实际上并不改变来源（不论是否同源）的shape
        self.qkv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)  # qkv1==》query,qkv2==>key,value
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    # 因为是计算交叉注意力，所以需要传入两个不同的来源的input
    def forward(self, x, guided,mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        # qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = self.qkv1(x).reshape(B_, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.qkv2(guided).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # @符号表示矩阵乘法

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x



class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        fused_window_process (bool, optional): If True, use one kernel to fused window shift & window partition for acceleration, similar for the reversed part. Default: False
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 fused_window_process=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:  # window size基本不可能大于input resolution
            # if window size is larger than input resolution, we don't partition windows  
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)  # 隐藏层放大了4倍
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 困难之处：attn mask如何应用在交叉注意力上，需要在意吗？
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            # 为了计算同样数量的windows的self-attention，故而需要进行偏移
            # 不同size的切块索引是不一样的
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt          # 标记分块索引
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)   # 这里有涉及到广播机制
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))  # 将mask的值设置为-100，经过softmax之后，都将变成0
        # 如果没有shift，那么则attention，不需要进行mask
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)
        self.fused_window_process = fused_window_process

    def forward(self, x,guided):
        H, W = self.input_resolution
        B, L, C = x.shape   # 因为在embedding的时候，已经摊平
        
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        guided = self.norm1(guided)
        guided = guided.view(B, H, W, C)


        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

            shifted_guided = torch.roll(guided, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            # partition windows
            guided_windows = window_partition(shifted_guided, self.window_size)  # nW*B, window_size, window_size, C

        else:
            shifted_x = x
            # partition windows
            x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C

            guided_windows = window_partition(guided, self.window_size)  # nW*B, window_size, window_size, C

        # 为什么不在partition里面直接view成 BLC的形式？。。。
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        guided_windows = guided_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C


        # W-MSA/SW-MSA      窗口多头自注意力，移动窗口多头自注意力
        attn_windows = self.attn(x_windows,guided_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows     恢复
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)

        # reverse cyclic shift
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = shifted_x
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)

    
        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # drop_path 是不同深度的dropout层
        vardict = locals()
   

        return x

class Spatial_aligner(nn.Module):
    def __init__(self,in_channel=192,out_channel=192,input_resolution=(224,224)) -> None:
        super().__init__()
        self.window_size = 4
        self.patch_size = 2 
        self.input_resolution = input_resolution
        self.embed_dim = 96
        self.patch_embeding1 = PatchEmbed(img_size=input_resolution, patch_size=self.patch_size, in_chans=in_channel, embed_dim=self.embed_dim)  # 如果没有使用shift窗口，那么没有获取全局特征？但是好处是：img_size能够做到比较随机？
        self.patch_embeding2 = PatchEmbed(img_size=input_resolution, patch_size=self.patch_size, in_chans=in_channel, embed_dim=self.embed_dim)
        self.blocks = nn.ModuleList([
                    SwinTransformerBlock(dim=96, num_heads=3, window_size=4,
                                        input_resolution=(self.input_resolution[0]//self.patch_size,self.input_resolution[1]//self.patch_size), # 元组不可以直接除法
                                        shift_size=0 if (i % 2 == 0) else 4 // 2)
                    for i in range(2)])  # range(depth) 每块的depth是不同的

        # self.recovery = deconv(self.embed_dim,out_channels=out_channel,kernel_size=self.patch_size,stride=self.patch_size)  
        self.recovery = nn.ConvTranspose2d(self.embed_dim,out_channel,kernel_size=self.patch_size,stride=self.patch_size)  

    def forward(self,x,guided):
        B,C,H,W = x.shape
        # embed_x = self.patch_embeding1(x)            # BLC
        # embed_guided = self.patch_embeding2(guided)  # BLC
        # for layer in self.blocks:
        #     out = layer(embed_x,embed_guided)  # layer1 完全被中断，因此每次进入，其实都没有进行梯度回传，也就是减少一个模块，是否有所影响？
        out = self.patch_embeding1(x)            # BLC
        embed_guided = self.patch_embeding2(guided)  # BLC
        for layer in self.blocks:
            out = layer(out,embed_guided)

        # 因为最后一层是MLP，故shape应该为 BLC  # 因为需要进行复原,为了避免padding操作，需要要求图片的shape为偶数
        out = out.contiguous().view(B,self.embed_dim,H//self.patch_size,W//self.patch_size) 
        out = self.recovery(out)
     

        return out

# decoder暂时不合并，因为没有什么功能是一定要
class Master_decoder(nn.Module):
    def __init__(self,N=192,M=192,channel=64*2,width=224,height=224,first_stride=2,master_chl=3) -> None:
        super().__init__()
        self.encoder_first_stride = first_stride
        width //= self.encoder_first_stride
        height //= self.encoder_first_stride

        self.g_s_conv1 = deconv(M, N, kernel_size=5, stride=2)
        self.g_s_gdn1 = GDN(N, inverse=True)
        self.sp_aligner1 = Spatial_aligner(input_resolution=(width//4,height//4))

        self.g_s_conv2 = deconv(2*N, N, kernel_size=5, stride=2)
        self.g_s_gdn2 = GDN(N, inverse=True)
        self.sp_aligner2 = Spatial_aligner(input_resolution=(width//2,height//2))

        self.g_s_conv3 = deconv(2*N, N, kernel_size=5, stride=2)
        self.g_s_gdn3 = GDN(N, inverse=True)
        self.sp_aligner3 = Spatial_aligner(input_resolution=(width//1,height//1))

        self.g_s_conv4 = deconv(2*N, channel, kernel_size=5, stride=self.encoder_first_stride)  # 通道数的变化，原论文有bug，但是只要计算好即可

        # 如果是master_chl == 1,那么需要进行特征对齐，就必须在进行一次下采样
        self.master_chl = master_chl
        # print('decoder:',master_chl)
        if master_chl == 1:
            self.downsample1 = conv(N, N, kernel_size=5, stride=2)
            self.downsample2 = conv(N, N, kernel_size=5, stride=2)
            self.downsample3 = conv(N, N, kernel_size=5, stride=2)


    def forward(self,x, guide_hidden):
        g_s_g1 = guide_hidden["gs1"]
        g_s_g2 = guide_hidden["gs2"]
        g_s_g3 = guide_hidden["gs3"]

        # print("x",x.shape)
        # print("g_s_g1",g_s_g1.shape)
        # print("g_s_g2",g_s_g2.shape)
        # print("g_s_g3",g_s_g3.shape)
        if self.master_chl == 1:
            g_s_g1=self.downsample1(g_s_g1)
            g_s_g2=self.downsample2(g_s_g2)
            g_s_g3=self.downsample3(g_s_g3)


        out1 = self.g_s_conv1(x)
        out1 = self.g_s_gdn1(out1)
        identity1 = out1
        out1 = self.sp_aligner1(out1,g_s_g1)  

        out1 = torch.cat([out1,identity1],dim=1)

        out2 = self.g_s_conv2(out1)           # 注意，这里由于拼接，in-channel会发生变化
        out2 = self.g_s_gdn2(out2)
        identity2 = out2
        out2 = self.sp_aligner2(out2,g_s_g2)  
        out2 = torch.cat([out2,identity2],dim=1)

        out3 = self.g_s_conv3(out2)           # 注意，这里由于拼接，in-channel会发生变化
        out3 = self.g_s_gdn3(out3)
        identity3 = out3
        out3 = self.sp_aligner3(out3,g_s_g3)  
        out3 = torch.cat([out3,identity3],dim=1)

        out = self.g_s_conv4(out3)
        
        return {"x_feature_hat":out}


# ==============================================Decoder==========================================


# ==============================================Master_compresser==========================================
# ==============================================Master_compresser==========================================

def myprint(var,vardict,type=1):
    name = [k for k,v in vardict.items() if v is var][0]  
    if type == 0 :
        print(name,var)
    else:
        print(name,var.shape)


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

# scale_table:0.11-256  [64]
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))  # exp和log是逆运算，范围一样，相对来说比较平滑

class Master_compresser(MeanScaleHyperprior):
    def __init__(self,width=256,height=256,channel=3,N=192,M=192) -> None:
        super().__init__(M,M)
        master_chl = 3 
        guided_chl = 1 
        master_stride=2
        guided_stride = 1
        if channel == 1:
            master_chl = 1
            guided_chl = 3
            guided_stride = 2
            master_stride = 1

        self.fencoder1 = Feature_encoder(in_channel=master_chl,out_channel=64,stride=master_stride) # 加载对应数据集的时候要注意对应关系
        self.fencoder2 = Feature_encoder(in_channel=guided_chl,out_channel=64,stride=guided_stride)
        

        self.ch_aligner=Channel_aligner()  # 64->64
        # self.encoder = Master_encoder(N=192,M=192,channel=64*2)

        # master_encoder直接从这里实现了
        self.g_a = nn.Sequential(
            conv(64*2, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),  # in,out,kernel_size
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),  # 2M
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        # self.entropy_bottleneck = EntropyBottleneck(M)  # 需要关注继承关系的复杂程度
        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)
        self.decoder = Master_decoder(N=192,M=192,channel=64*2,width=width,height=height,first_stride=2,master_chl=master_chl)

        self.fdecoder = Feature_decoder(in_channel=64*3,out_channel=master_chl,stride=master_stride)

    def forward(self,x,guided_hat,guided_hidden):
        # print("Mc.forward",x.shape,guided_hat.shape)
        x_feature = self.fencoder1(x)
        guided_feature = self.fencoder2(guided_hat)
        
        guided_align,beta,gamma = self.ch_aligner(x_feature,guided_feature)

        input = torch.cat([x_feature,guided_align],dim=1)
        
        # res_enc = self.encoder(input)
        y = self.g_a(input)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        params = self.h_s(z_hat)

        y_hat = self.gaussian_conditional.quantize(
            y, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y_hat)  
        gaussian_params = self.entropy_parameters(   
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)


        res_dec = self.decoder(y_hat,guided_hidden)
        
        feature_dec = torch.cat([res_dec["x_feature_hat"],guided_align],dim=1) 
        out = self.fdecoder(feature_dec)
        
        # y_hat = res_enc["y_hat"]
        # x_hat = res_dec["x_feature_hat"]
        # vardict = locals()
        # myprint(x_feature,vardict)
        # myprint(guided_feature,vardict)
        # myprint(guided_align,vardict)
        # myprint(beta,vardict)
        # myprint(gamma,vardict)
        # myprint(y_hat,vardict)
        # myprint(x_hat,vardict)
        # myprint(out,vardict)

        
        return {
            "x_hat": out,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x, guided_hat):
        print("x.shape",x.shape)
        print("guided_hat.shape",guided_hat.shape)
       
        x_feature = self.fencoder1(x)
        guided_feature = self.fencoder2(guided_hat)
        guided_align,beta,gamma = self.ch_aligner(x_feature,guided_feature)
        input = torch.cat([x_feature,guided_align],dim=1)

        y = self.g_a(input)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)            # h_s,还需要和context，以及entropy_parameter共用才行

        s = 4                               # scaling factor between z and y
        kernel_size = 5                     # context prediction kernel size
        padding = (kernel_size - 1) // 2    # padding = 2   

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):          # 批次中的每一个
            string = self._compress_ar(
                y_hat[i: i + 1],            # 有冒号，只是为了保证维度
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],"gamma":gamma,"beta":beta}  # gamma和beta是无损压缩

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        masked_weight = self.context_prediction.weight * self.context_prediction.mask   # context实际上由一个mask卷积核组成，而全通道共享一个卷积核
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]          
                ctx_p = F.conv2d(                                                       # 一个mask_convolution卷积的结果就是一个数值
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]                                    
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)          

                y_crop = y_crop[:, :, padding, padding]                                 # 取出所需要编码的像素，其他部分都只是padding
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)  
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())                             # extend会将添加项，修改成可迭代类型；append则是直接原封不动地添加
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    # 更新scale_table和gs.update
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()  # 0.11~256，64格
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)  # 更新真实的高斯熵编码模型
        updated |= super().update(force=force)                                            # 更新entropy_bottleneck模型
        return updated

    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        # 是否包含了子模块中的模块？
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)  # 只有entropyBottleneck模块需要
        )
        return aux_loss

    # 编解码其实是可以继承的,如果直接交给master去继承，是会比较方便，但是master继承之后的参数量是否会变大？或者不要调用super().__init__()?先跑起来，再来考虑优化
    def decompress(self, out_net,out_net_guided):
        strings = out_net["strings"]
        shape = out_net["shape"]
        beta = out_net["beta"]
        gamma = out_net["gamma"] 
        guided_hidden = out_net_guided['hidden']  # 用来辅助解码
        guided_align = self.fencoder2(out_net_guided['x_hat'])  # 通道对齐
        guided_align = gamma * guided_align + beta
        
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        
        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        res_dec = self.decoder(y_hat,guided_hidden)
        feature_dec = torch.cat([res_dec["x_feature_hat"],guided_align],dim=1) 
        x_hat = self.fdecoder(feature_dec).clamp_(0, 1)
        return {"x_hat": x_hat}

    def _decompress_ar(
            self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)  # encoder_with_indexes = set_stream+decode_stream

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv
    
    # checkpoint不需要层层包装，自己可以写死
    # 可以直接进行实例化，然后torch.load(load_state())即可

    # @classmethod
    # def from_state_dict(cls, state_dict, channel=3):
    #     """Return a new model instance from `state_dict`."""
    #     N = state_dict["g_a.0.weight"].size(0)
    #     M = state_dict["g_a.6.weight"].size(0)
    #     net = cls(N, M, channel)
    #     net.load_state_dict(state_dict)
    #     return net

# ==============================================Master_compresser==========================================


# ==============================================Guided_compresser==========================================
# ==============================================Guided_compresser==========================================
# Gudied的 encoder和decoderr很相似，可以考虑合并，暂时先写完能够训练就行
class Encoder1(nn.Module):
    def __init__(self, N, M,channel=1,first_stride=2,**kwargs):
        super().__init__()
        self.g_a_conv1 = conv(channel, N, kernel_size=5, stride=first_stride)
        self.g_a_gdn1 = GDN(N)
        self.g_a_conv2 = conv(N, N, kernel_size=5, stride=2)
        self.g_a_gdn2 = GDN(N)
        self.g_a_conv3 = conv(N, N, kernel_size=5, stride=2)
        self.g_a_gdn3 = GDN(N)
        self.g_a_conv4 = conv(N, M, kernel_size=5, stride=2)

    def forward(self, x):  # 此处返回的输出层在GDN之前还是之后需要再确定
        self.g_a_c1 = self.g_a_conv1(x)  # Tensor
        self.g_a_g1 = self.g_a_gdn1(self.g_a_c1)
        self.g_a_c2 = self.g_a_conv2(self.g_a_g1)  # Tensor
        self.g_a_g2 = self.g_a_gdn2(self.g_a_c2)
        self.g_a_c3 = self.g_a_conv3(self.g_a_g2)  # Tensor
        self.g_a_g3 = self.g_a_gdn3(self.g_a_c3)
        self.g_a_c4 = self.g_a_conv4(self.g_a_g3)  # Tensor
        self.y = self.g_a_c4
        return self.y, self.g_a_g1, self.g_a_g2, self.g_a_g3

    

# 相当于g_s部分
class Decoder1(nn.Module):
    def __init__(self, N, M, channel=1,first_stride=2,**kwargs):
        super().__init__()
        self.g_s_conv1 = deconv(M, N, kernel_size=5, stride=2)
        self.g_s_gdn1 = GDN(N, inverse=True)
        self.g_s_conv2 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_gdn2 = GDN(N, inverse=True)
        self.g_s_conv3 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_gdn3 = GDN(N, inverse=True)
        self.g_s_conv4 = deconv(N, channel, kernel_size=5, stride=first_stride)

    def forward(self, y_hat):
        # self.x_hat = self.g_s(self.y_hat)
        self.g_s_c1 = self.g_s_conv1(y_hat)  # Tensor
        self.g_s_g1 = self.g_s_gdn1(self.g_s_c1)
        self.g_s_c2 = self.g_s_conv2(self.g_s_g1)  # Tensor
        self.g_s_g2 = self.g_s_gdn2(self.g_s_c2)
        self.g_s_c3 = self.g_s_conv3(self.g_s_g2)  # Tensor
        self.g_s_g3 = self.g_s_gdn3(self.g_s_c3)
        self.g_s_c4 = self.g_s_conv4(self.g_s_g3)  # Tensor
        self.x_hat = self.g_s_c4
        return self.x_hat, self.g_s_g1, self.g_s_g2, self.g_s_g3

class Guided_compresser(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192,channel=1,first_stride=2, **kwargs):
        super().__init__(N=N, M=M, **kwargs)
        self.first_stride=first_stride
        self.enc1 = Encoder1(N, M,channel,self.first_stride)
        self.dec1 = Decoder1(N, M,channel,self.first_stride)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(N, N, stride=2, kernel_size=5),
        )

        self.h_s = nn.Sequential(
            deconv(N, M, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2, stride=2, kernel_size=5),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(M * 12 // 3, M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 10 // 3, M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(M * 8 // 3, M * 6 // 3, 1),
        )

        self.context_prediction = MaskedConv2d(
            M, 2 * M, kernel_size=5, padding=2, stride=1
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):  # x for depth map; y for guided rgb map
        # for rgb
        y1, g_a_g1, g_a_g2, g_a_g3 = self.enc1(x)
        # y1 = self.g_a(x)
        z1 = self.h_a(y1)

        z1_hat, z1_likelihoods = self.entropy_bottleneck(z1)
        params = self.h_s(z1_hat)

        y1_hat = self.gaussian_conditional.quantize(
            y1, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y1_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y1_likelihoods = self.gaussian_conditional(y1, scales_hat, means=means_hat)
        # x1_hat = self.g_s(y1_hat)
        x1_hat, g_s_g1, g_s_g2, g_s_g3 = self.dec1(y1_hat)


        return {
            "x_hat": x1_hat,
            "likelihoods": {"y": y1_likelihoods, "z": z1_likelihoods},
            "hidden": {"ga1": g_a_g1, "ga2": g_a_g2, "ga3": g_a_g3, "gs1": g_s_g1, "gs2": g_s_g2, "gs3": g_s_g3},
        }

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y, g_a_g1, g_a_g2, g_a_g3 = self.enc1(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        y_hat = F.pad(y, (padding, padding, padding, padding))

        y_strings = []
        for i in range(y.size(0)):
            string = self._compress_ar(
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )
            y_strings.append(string)

        # 为什么要返回这么多中间值？供给depth网络使用。
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:],
                "hidden": {'ga1': g_a_g1, 'ga2': g_a_g2, 'ga3': g_a_g3}}

    def _compress_ar(self, y_hat, params, height, width, kernel_size, padding):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []

        # Warning, this is slow...
        # TODO: profile the calls to the bindings...
        masked_weight = self.context_prediction.weight * self.context_prediction.mask
        for h in range(height):
            for w in range(width):
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    masked_weight,
                    bias=self.context_prediction.bias,
                )

                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                gaussian_params = gaussian_params.squeeze(3).squeeze(2)
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)

                y_crop = y_crop[:, :, padding, padding]
                y_q = self.gaussian_conditional.quantize(y_crop, "symbols", means_hat)
                y_hat[:, :, h + padding, w + padding] = y_q + means_hat

                symbols_list.extend(y_q.squeeze().tolist())
                indexes_list.extend(indexes.squeeze().tolist())

        encoder.encode_with_indexes(
            symbols_list, indexes_list, cdf, cdf_lengths, offsets
        )

        string = encoder.flush()
        return string

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        # FIXME: we don't respect the default entropy coder and directly call the
        # range ANS decoder

        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        params = self.h_s(z_hat)

        s = 4  # scaling factor between z and y
        kernel_size = 5  # context prediction kernel size
        padding = (kernel_size - 1) // 2

        y_height = z_hat.size(2) * s
        y_width = z_hat.size(3) * s

        # initialize y_hat to zeros, and pad it so we can directly work with
        # sub-tensors of size (N, C, kernel size, kernel_size)
        y_hat = torch.zeros(
            (z_hat.size(0), self.M, y_height + 2 * padding, y_width + 2 * padding),
            device=z_hat.device,
        )

        for i, y_string in enumerate(strings[0]):
            self._decompress_ar(
                y_string,
                y_hat[i: i + 1],
                params[i: i + 1],
                y_height,
                y_width,
                kernel_size,
                padding,
            )

        y_hat = F.pad(y_hat, (-padding, -padding, -padding, -padding))
        # x_hat = self.g_s(y_hat)# .clamp_(0, 1)
        x_hat, g_s_g1, g_s_g2, g_s_g3 = self.dec1(y_hat)
        return {"x_hat": x_hat.clamp(0, 1), "hidden": {'gs1': g_s_g1, 'gs2': g_s_g2, 'gs3': g_s_g3}}

    def _decompress_ar(
            self, y_string, y_hat, params, height, width, kernel_size, padding
    ):
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.tolist()
        offsets = self.gaussian_conditional.offset.tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        # Warning: this is slow due to the auto-regressive nature of the
        # decoding... See more recent publication where they use an
        # auto-regressive module on chunks of channels for faster decoding...
        for h in range(height):
            for w in range(width):
                # only perform the 5x5 convolution on a cropped tensor
                # centered in (h, w)
                y_crop = y_hat[:, :, h: h + kernel_size, w: w + kernel_size]
                ctx_p = F.conv2d(
                    y_crop,
                    self.context_prediction.weight,
                    bias=self.context_prediction.bias,
                )
                # 1x1 conv for the entropy parameters prediction network, so
                # we only keep the elements in the "center"
                p = params[:, :, h: h + 1, w: w + 1]
                gaussian_params = self.entropy_parameters(torch.cat((p, ctx_p), dim=1))
                scales_hat, means_hat = gaussian_params.chunk(2, 1)

                indexes = self.gaussian_conditional.build_indexes(scales_hat)
                rv = decoder.decode_stream(
                    indexes.squeeze().tolist(), cdf, cdf_lengths, offsets
                )
                rv = torch.Tensor(rv).reshape(1, -1, 1, 1)
                rv = self.gaussian_conditional.dequantize(rv, means_hat)

                hp = h + padding
                wp = w + padding
                y_hat[:, :, hp: hp + 1, wp: wp + 1] = rv

# ==============================================Guided_compresser==========================================

        
import random
# width,height为bounding box
def random_crop(img, guided, width, height):  
    assert img.shape[0] >= height
    assert img.shape[1] >= width
    assert img.shape[0] == 2 * guided.shape[0]
    assert img.shape[1] == 2 * guided.shape[1]
    x = random.randint(0, guided.shape[1] - width)
    y = random.randint(0, guided.shape[0] - height)
    img = img[2*y:2*(y+height), 2*x:2*(x+width)]  # 卷积通过stride会将形状转换成一样的
    guided = guided[y:y+height, x:x+width]
    return img, guided

# 仅在example文件夹下可以使用
if __name__ == '__main__':
    # 可以统计一下参数，看看要不要进行random_crop，有多大的影响？
    # 复现的时候数据增强这种trick，是否有很大的影响?

    # 不同shape的都可以参与训练，但是大尺寸的训练的慢。
    # 如果同时训练，显存是否足够？应该是足够的。
    # 如果不同时训练，因为不同尺寸，可能会对应不上。
    
    device = 'cpu'
    guided = torch.randn(8,1,640,512).to(device)     # 640*512
    x = torch.randn(8,3,1280,1024).to(device)        # 1280*1024

    # guided = torch.randn(8,1,224,224)     # 224/256
    # x = torch.randn(8,3,224*2,224*2)

    # guided = torch.randn(8,1,256,256)     # 224/256
    # x = torch.randn(8,3,256*2,256*2)

    b,h,width,height = guided.shape

    Guider = Guided_compresser().to(device)
    Master = Master_compresser(width=width,height=height).to(device)

    guider_out = Guider(guided)
    Master(x,guided,guider_out["hidden"])

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    #  print(Master.modules)
    summary(Guider,input_data=[guided])
    summary(Master,input_data=[x,guided,guider_out['hidden']])




    


    
