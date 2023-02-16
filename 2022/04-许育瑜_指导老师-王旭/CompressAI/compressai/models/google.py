# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.layers import GDN, MaskedConv2d

from .utils import conv, deconv, update_registered_buffers
import torchsnooper

__all__ = [
    "CompressionModel",
    "FactorizedPrior",
    "ScaleHyperprior",
    "MeanScaleHyperprior",
    "JointAutoregressiveHierarchicalPriors",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
]


# 对所有（至少一个）熵模型的封装
class CompressionModel(nn.Module):
    """Base class for constructing an auto-encoder with at least one entropy
    bottleneck module.

    Args:
        entropy_bottleneck_channels (int): Number of channels of the entropy
            bottleneck
    """

    # 根据通道数，初始化熵瓶颈模型
    def __init__(self, entropy_bottleneck_channels, init_weights=None):
        super().__init__()
        self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

        if init_weights is not None:
            warnings.warn(
                "init_weights was removed as it was never functional",
                DeprecationWarning,
            )

    # 辅助损失：用于真正编码时的概率分布损失计算
    def aux_loss(self):
        """Return the aggregated loss over the auxiliary entropy bottleneck
        module(s).
        """
        aux_loss = sum(
            m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck)
        )
        return aux_loss

    def forward(self, *args):
        raise NotImplementedError()

    # 更新熵瓶颈模型的F值
    def update(self, force=False):
        """Updates the entropy bottleneck(s) CDF values.  # 更新熵瓶颈模型的F值

        Needs to be called once after training   # 训练完调用更新
        to be able to later perform the evaluation with an actual entropy coder.  # 能够评价一个熵编码模型

        Args:
            force (bool): overwrite previous values (default: False)  # 是否重写cdf值
 
        Returns:
            updated (bool): True if one of the EntropyBottlenecks was updated.  # 是否完成了更新

        """
        updated = False
        # self.children()只包括网络模块的第一代儿子模块，而self.modules()包含网络模块的自己本身和所有后代模块。
        for m in self.children():  
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)                  # 更新：entropy_bottleneck
            updated |= rv                               # 完成更新之后，updated返回true
        return updated

    # 加载并更新熵瓶颈模型的cdf值
    def load_state_dict(self, state_dict):
        # Dynamically update the entropy bottleneck buffers related to the CDFs  # 动态更新关于cdf的熵瓶颈buffer
        update_registered_buffers(
            self.entropy_bottleneck,  # module 
            "entropy_bottleneck",  # module name
            ["_quantized_cdf", "_offset", "_cdf_length"],  # buffer name
            state_dict,  # state_dict
        )
        super().load_state_dict(state_dict)  # 将 state_dict 中的 parameters 和 buffers 复制到此 module 及其子节点中


# factor
class FactorizedPrior(CompressionModel):
    r"""Factorized Prior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_, Int Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    # 初始化网络结构
    def __init__(self, N, M, channel=3, **kwargs):
        super().__init__(entropy_bottleneck_channels=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(channel, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, channel),
        )

        self.N = N
        self.M = M

    # 下采样16倍
    @property
    def downsampling_factor(self) -> int:
        return 2 ** 4

    # 计算出y,y_hat,y_likelihoods,x_hat
    def forward(self, x):
        y = self.g_a(x)
        y_hat, y_likelihoods = self.entropy_bottleneck(y)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {
                "y": y_likelihoods,
            },
        }

    # 类方法：根据state，创建一个新的实例
    @classmethod
    def from_state_dict(cls, state_dict, channel=3):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)  # 怪不得看起来，不需要传入通道，原来在这里【而且因为在update文件中，已经完成了重新命名】
        net = cls(N, M, channel=channel)
        net.load_state_dict(state_dict)
        return net

    # 将经过g_a的结果，压缩成二进制
    def compress(self, x):
        y = self.g_a(x)
        y_strings = self.entropy_bottleneck.compress(y)
        return {"strings": [y_strings], "shape": y.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 1
        y_hat = self.entropy_bottleneck.decompress(strings[0], shape)
        x_hat = self.g_s(y_hat).clamp_(0, 1)  # 为什么解压成0-1之间？因为输入是进行了归一化处理的，那么输出自然也需要满足(0,1)
        return {"x_hat": x_hat}


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64

# scale_table:0.11-256  [64]
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))  # exp和log是逆运算，范围一样，相对来说比较平滑


# factor==>scale
class ScaleHyperprior(CompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    # @torchsnooper.snoop()
    def __init__(self, N, M, channel=3, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            conv(channel, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, N),
            GDN(N),
            conv(N, M),
        )

        self.g_s = nn.Sequential(
            deconv(M, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, channel),
        )

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    # 下采样16->64倍
    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    # @torchsnooper.snoop()
    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        # print('forward feature:',z_hat[1,:10,1,1])
        scales_hat = self.h_s(z_hat)
        print('scale_hat.shape',scales_hat.shape)
        print('y.shape',y.shape)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    # 与基础模型不同的是，多了一个scale属性
    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict, channel=3):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M, channel=channel)
        net.load_state_dict(state_dict)
        return net

    # 更新scale_table和gs.update
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()  # 0.11~256，64格
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)  # 更新真实的高斯熵编码模型
        updated |= super().update(force=force)                                            # 更新entropy_bottleneck模型
        return updated

    # 与基础模型不同的是，z_hat使用entropy_bottleneck,但是y压缩成二进制，采用的是gaussian_condiitonal
    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)                                   # z_hat==>h_s==>scales_hat 解码出来，应当是特征的每个像素点，都对应着indexes[64*8*8]
        indexes = self.gaussian_conditional.build_indexes(scales_hat)  # conditonal是在entroy model中调用的，这里的build_indexes是在这里调用的
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)                                    # 为什么不直接压缩scale？因为中间结果是进行了下采样的，压缩所需比特更少
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


# scale+mean
class MeanScaleHyperprior(ScaleHyperprior):
    r"""Scale Hyperprior with non zero-mean Gaussian conditionals from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N, M, channel=3, **kwargs):
        super().__init__(N, M, channel, **kwargs)

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, M),
            nn.LeakyReLU(inplace=True),
            deconv(M, M * 3 // 2),
            nn.LeakyReLU(inplace=True),
            conv(M * 3 // 2, M * 2, stride=1, kernel_size=3),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)  
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)  # \mu,\sigma
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}


# scale+mean+context[不能使用gpu]
# 因为context对上下文有依赖，因此需要串行计算
class JointAutoregressiveHierarchicalPriors(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, channel=3, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.g_a = nn.Sequential(
            conv(channel, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, N, kernel_size=5, stride=2),
            GDN(N),
            conv(N, M, kernel_size=5, stride=2),
        )

        self.g_s = nn.Sequential(
            deconv(M, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, N, kernel_size=5, stride=2),
            GDN(N, inverse=True),
            deconv(N, channel, kernel_size=5, stride=2),
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

    # 下采样16->64倍
    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

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
        # 熵模型，forward也只是计算概率和量化而已，这里外面已经经过量化了
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        # 为什么这样返回？因为计算率损失的时候，仅仅需要用到概率；失真损失的时候，仅仅需要x_hat
        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict, channel=3):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M, channel)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        # 不支持并行处理，所以运行速度就会很限制
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
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

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

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
        # 所有通道，每个位置，都有一个对应的分布N(\mu,\sigma^2)
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
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

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


### =======================jointly train=========================
class Encoder1(nn.Module):
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.g_a_conv1 = conv(3, N, kernel_size=5, stride=2)
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
    def __init__(self, N, M, **kwargs):
        super().__init__()
        self.g_s_conv1 = deconv(M, N, kernel_size=5, stride=2)
        self.g_s_gdn1 = GDN(N, inverse=True)
        self.g_s_conv2 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_gdn2 = GDN(N, inverse=True)
        self.g_s_conv3 = deconv(N, N, kernel_size=5, stride=2)
        self.g_s_gdn3 = GDN(N, inverse=True)
        self.g_s_conv4 = deconv(N, 3, kernel_size=5, stride=2)

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


# rgb部分，使用了encoder1，和decoder1，然后返回了gdn之后的中间结果（forward、compress、decompress）
class JointAutoregressiveHierarchicalPriors_R(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.enc1 = Encoder1(N, M)
        self.dec1 = Decoder1(N, M)

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

    @classmethod
    def from_state_dict(cls, state_dict):  # 因为有super()的存在，所以这样的命名法是可以使用的。同时，net应该是直接写死的，所以不用管channel
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

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

        # 为什么要返回这么多中间值？
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


class JointAutoregressiveHierarchicalPriors_D(MeanScaleHyperprior):
    r"""Joint Autoregressive Hierarchical Priors model from D.
    Minnen, J. Balle, G.D. Toderici: `"Joint Autoregressive and Hierarchical
    Priors for Learned Image Compression" <https://arxiv.org/abs/1809.02736>`_,
    Adv. in Neural Information Processing Systems 31 (NeurIPS 2018).

    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, N=192, M=192, **kwargs):
        super().__init__(N=N, M=M, **kwargs)

        self.pic2_g_a_conv1 = conv(1, N)
        self.pic2_g_a_gdn1 = GDN(N)
        self.pic2_g_a_conv2 = conv(2 * N, N)
        self.pic2_g_a_gdn2 = GDN(N)
        self.pic2_g_a_conv3 = conv(2 * N, N)
        self.pic2_g_a_gdn3 = GDN(N)
        self.pic2_g_a_conv4 = conv(2 * N, M)
        #
        self.pic2_g_s_conv1 = deconv(M, N)
        self.pic2_g_s_gdn1 = GDN(N, inverse=True)
        self.pic2_g_s_conv2 = deconv(2 * N, N)
        self.pic2_g_s_gdn2 = GDN(N, inverse=True)
        self.pic2_g_s_conv3 = deconv(2 * N, N)
        self.pic2_g_s_gdn3 = GDN(N, inverse=True)
        self.pic2_g_s_conv4 = deconv(2 * N, 1)

        self.tran_conv1 = conv(2 * N, N, stride=1)
        self.tran_conv2 = conv(2 * N, N, stride=1)
        self.tran_conv3 = conv(2 * N, N, stride=1)
        self.tran_conv4 = conv(2 * N, N, stride=1)
        self.tran_conv5 = conv(2 * N, N, stride=1)
        self.tran_conv6 = conv(2 * N, N, stride=1)

        # self.deconv1 = deconv(N,N)
        # self.deconv2 = deconv(N,N)
        # self.conv1 = conv(N,N)
        # self.conv2 = conv(N,N)

        # self.max_pool = nn.MaxPool2d(kernel_size=2,stride=2,return_indices=True)
        # self.max_unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #我觉得是会影响到重建的效果的

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

        self.r1 = nn.ReLU()
        self.r2 = nn.ReLU()
        self.r3 = nn.ReLU()
        self.r4 = nn.ReLU()
        self.r5 = nn.ReLU()
        self.r6 = nn.ReLU()

        self.attention1 = ESA(N)
        self.attention2 = ESA(N)
        self.attention3 = ESA(N)
        self.attention4 = ESA(N)
        self.attention5 = ESA(N)
        self.attention6 = ESA(N)

        # 11.18 将边缘提取的方式改为conv+relu
        self.eg_ext1 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext2 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext3 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext4 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext5 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext6 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext7 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext8 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext9 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext10 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext11 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))
        self.eg_ext12 = nn.Sequential(
            nn.Conv2d(N, N, stride=1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True))

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x, hidden):  # x for depth map; y for guided rgb map

        # guided feature
        g_a_g1 = hidden["ga1"]
        g_a_g2 = hidden["ga2"]
        g_a_g3 = hidden["ga3"]
        g_s_g1 = hidden["gs1"]
        g_s_g2 = hidden["gs2"]
        g_s_g3 = hidden["gs3"]

        x = self.pic2_g_a_conv1(x)
        x = self.pic2_g_a_gdn1(x)

        # level 1
        eg1 = self.eg_ext1(x)  # 提取depth特征
        g_a_g1 = self.eg_ext2(g_a_g1)  # 提取color特征
        g_a_g1 = self.tran_conv1(torch.cat((eg1, g_a_g1), dim=-3))  # fusion
        # g_a_g1 = self.r1(g_a_g1)
        g_a_g1 = self.attention1(g_a_g1)  # ESA attention
        # ga1_1 = self.conv1(g_a_g1) # h*w -> 1/2 (h*w)
        # ga1_1 = self.max_pool(g_a_g1)

        x = self.pic2_g_a_conv2(torch.cat((x, g_a_g1), dim=-3))
        x = self.pic2_g_a_gdn2(x)

        # level 2
        eg2 = self.eg_ext3(x)
        g_a_g2 = self.eg_ext4(g_a_g2)
        g_a_g2 = self.tran_conv2(torch.cat((eg2, g_a_g2), dim=-3))
        # g_a_g2 = self.r2(g_a_g2)
        g_a_g2 = self.attention2(g_a_g2)
        # g_a_g2 = self.down_ch1(torch.cat((ga1_1, g_a_g2), dim=-3)) # multi-scale fusion N + N -> N
        # ga2_1 = self.conv2(g_a_g2) # 1/2 (h*w) -> 1/4 (h*w)
        # ga2_1 = self.max_pool(g_a_g2)

        x = self.pic2_g_a_conv3(torch.cat((x, g_a_g2), dim=-3))
        x = self.pic2_g_a_gdn3(x)

        # level 3
        eg3 = self.eg_ext5(x)
        g_a_g3 = self.eg_ext6(g_a_g3)
        g_a_g3 = self.tran_conv3(torch.cat((eg3, g_a_g3), dim=-3))
        # g_a_g3 = self.r3(g_a_g3)
        g_a_g3 = self.attention3(g_a_g3)
        # g_a_g3 = self.down_ch2(torch.cat((ga2_1, g_a_g3), dim=-3))

        y2 = self.pic2_g_a_conv4(torch.cat((x, g_a_g3), dim=-3))
        z2 = self.h_a(y2)
        z2_hat, z2_likelihoods = self.entropy_bottleneck(z2)

        params = self.h_s(z2_hat)
        # y2_hat, y2_likelihoods = self.gaussian_conditional(y2, scales_hat)

        # for rgb
        # y1 = self.g_a(x)
        # z1 = self.h_a(y1)
        # z1_hat, z1_likelihoods = self.entropy_bottleneck(z1)
        # params = self.h_s(z1_hat)

        y2_hat = self.gaussian_conditional.quantize(
            y2, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y2_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y2_likelihoods = self.gaussian_conditional(y2, scales_hat, means=means_hat)

        # level 1
        x = self.pic2_g_s_conv1(y2_hat)
        x = self.pic2_g_s_gdn1(x)
        eg4 = self.eg_ext7(x)
        g_s_g1 = self.eg_ext8(g_s_g1)
        g_s_g1 = self.tran_conv4(torch.cat((eg4, g_s_g1), dim=-3))
        # g_s_g1 = self.r4(g_s_g1)
        g_s_g1 = self.attention4(g_s_g1)
        # gs1_1 = self.deconv1(g_s_g1)
        # gs1_1 = self.max_unpool(g_s_g1)

        # level 2
        x = self.pic2_g_s_conv2(torch.cat((x, g_s_g1), dim=-3))
        x = self.pic2_g_s_gdn2(x)
        eg5 = self.eg_ext9(x)
        g_s_g2 = self.eg_ext10(g_s_g2)
        g_s_g2 = self.tran_conv5(torch.cat((eg5, g_s_g2), dim=-3))
        # g_s_g2 = self.r5(g_s_g2)
        g_s_g2 = self.attention5(g_s_g2)
        # g_s_g2 = self.down_ch3(torch.cat((gs1_1, g_s_g2), dim=-3))
        # gs2_1 = self.deconv2(g_s_g2)
        # gs2_1 = self.max_unpool(g_s_g2)

        # level 3
        x = self.pic2_g_s_conv3(torch.cat((x, g_s_g2), dim=-3))
        x = self.pic2_g_s_gdn3(x)
        eg6 = self.eg_ext11(x)
        g_s_g3 = self.eg_ext12(g_s_g3)
        g_s_g3 = self.tran_conv6(torch.cat((eg6, g_s_g3), dim=-3))
        # g_s_g3 = self.r6(g_s_g3)
        g_s_g3 = self.attention6(g_s_g3)
        # g_s_g3 = self.down_ch4(torch.cat((gs2_1, g_s_g3), dim=-3))

        x2_hat = self.pic2_g_s_conv4(torch.cat((x, g_s_g3), dim=-3))
        # x1_hat = self.g_s(y2_hat)

        return {
            "x_hat": x2_hat,
            "likelihoods": {"y": y2_likelihoods, "z": z2_likelihoods},
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def compress(self, x):
        if next(self.parameters()).device != torch.device("cpu"):
            warnings.warn(
                "Inference on GPU is not recommended for the autoregressive "
                "models (the entropy coder is run sequentially on CPU)."
            )

        y = self.g_a(x)
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

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

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
        x_hat = self.g_s(y_hat)  # .clamp_(0, 1)
        return {"x_hat": x_hat}

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


def default_conv(in_channels, out_channels, kernel_size, stride=1, padding=None, bias=True, groups=1):
    if not padding and stride == 1:
        padding = kernel_size // 2
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)


class ESA(nn.Module):
    def __init__(self, n_feats, conv=default_conv):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = conv(n_feats, f, kernel_size=1)
        self.conv_f = conv(f, f, kernel_size=1)
        self.conv_max = conv(f, f, kernel_size=3, padding=1)
        self.conv2 = conv(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = conv(f, f, kernel_size=3, padding=1)
        self.conv3_ = conv(f, f, kernel_size=3, padding=1)
        self.conv4 = conv(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        f = x
        c1_ = (self.conv1(f))  # 1*1卷积，降低维度（减少计算复杂度）
        c1 = self.conv2(c1_)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)  # 增大感受野
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)  # upsample + 1*1卷积恢复通道数
        m = self.sigmoid(c4)  # 生成mask

        return x * m


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
