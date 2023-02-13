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

import torch.nn as nn
import torch

from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)

from .google import JointAutoregressiveHierarchicalPriors, JointAutoregressiveHierarchicalPriors_R, \
    JointAutoregressiveHierarchicalPriors_D,ESA
    
from compressai.models.utils import conv, deconv, update_registered_buffers


# “Learned Image Compression with Discretized Gaussian Mixture Likelihoods and Attention Modules 里面的anchor和attention模块”
class Cheng2020Anchor(JointAutoregressiveHierarchicalPriors):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, channel=3, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(channel, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
        )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),  # 这是进入parameter网络之前
        )

        self.g_s = nn.Sequential(
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, channel, 2),
        )

    @classmethod
    def from_state_dict(cls, state_dict, channel=3):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N, channel)
        net.load_state_dict(state_dict)
        return net


class Cheng2020Attention(Cheng2020Anchor):
    """Self-attention model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses self-attention, residual blocks with small convolutions (3x3 and 1x1),
    and sub-pixel convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, channel=3, **kwargs):
        super().__init__(N=N, **kwargs)

        self.g_a = nn.Sequential(
            ResidualBlockWithStride(channel, N, stride=2),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockWithStride(N, N, stride=2),
            ResidualBlock(N, N),
            conv3x3(N, N, stride=2),
            AttentionBlock(N),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            AttentionBlock(N),
            ResidualBlock(N, N),
            ResidualBlockUpsample(N, N, 2),
            ResidualBlock(N, N),
            subpel_conv3x3(N, channel, 2),
        )


# 这里的anchor，相比于waseda里面的，g_a和g_s放到了attn里面重写
class Cheng2020Anchor_R(JointAutoregressiveHierarchicalPriors_R):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

    @classmethod
    def from_state_dict(cls, state_dict): # 不需要调用，因为cheng2020-attn_R已经写死，直接调用load_state_dict
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class Cheng2020Attention_R(Cheng2020Anchor_R):
    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)
        self.enc = Encoder_cheng(N)
        self.dec = Decoder_cheng(N)

    def forward(self, x):  # x for depth map; y for guided rgb map
        # for rgb
        y1, g_a_g1, g_a_g2, g_a_g3 = self.enc(x)
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
        x1_hat, g_s_g1, g_s_g2, g_s_g3 = self.dec(y1_hat)

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

        y, g_a_g1, g_a_g2, g_a_g3 = self.enc(x)
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
        x_hat, g_s_g1, g_s_g2, g_s_g3 = self.dec(y_hat)
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


class Encoder_cheng(nn.Module):
    def __init__(self, N, **kwargs):
        super().__init__()
        self.res_stride1 = ResidualBlockWithStride(3, N, stride=2)
        self.res1 = ResidualBlock(N, N)
        self.res_stride2 = ResidualBlockWithStride(N, N, stride=2)
        self.atten1 = AttentionBlock(N)
        self.res2 = ResidualBlock(N, N)
        self.res_stride3 = ResidualBlockWithStride(N, N, stride=2)
        self.res3 = ResidualBlock(N, N)
        self.conv = conv3x3(N, N, stride=2)
        self.atten2 = AttentionBlock(N)

    def forward(self, x):
        x = self.res_stride1(x)
        hidden1 = self.res1(x)
        x = self.res_stride2(hidden1)
        x = self.atten1(x)
        hidden2 = self.res2(x)
        x = self.res_stride3(hidden2)
        hidden3 = self.res3(x)
        x = self.conv(x)
        x = self.atten2(x)
        return x, hidden1, hidden2, hidden3


class Decoder_cheng(nn.Module):
    def __init__(self, N, **kwargs):
        super().__init__()
        self.atten1 = AttentionBlock(N)
        self.res1 = ResidualBlock(N, N)
        self.res_stride1 = ResidualBlockUpsample(N, N, 2)
        self.res2 = ResidualBlock(N, N)
        self.res_stride2 = ResidualBlockUpsample(N, N, 2)
        self.atten2 = AttentionBlock(N)
        self.res3 = ResidualBlock(N, N)
        self.res_stride3 = ResidualBlockUpsample(N, N, 2)
        self.res4 = ResidualBlock(N, N)
        self.conv = subpel_conv3x3(N, 3, 2)

    def forward(self, x):
        x = self.atten1(x)
        x = self.res1(x)
        hidden1 = self.res_stride1(x)
        x = self.res2(hidden1)
        x = self.res_stride2(x)
        hidden2 = self.atten2(x)
        x = self.res3(hidden2)
        hidden3 = self.res_stride3(x)
        x = self.res4(hidden3)
        x = self.conv(x)
        return x, hidden1, hidden2, hidden3


class Cheng2020Anchor_D(JointAutoregressiveHierarchicalPriors_D):
    """Anchor model variant from `"Learned Image Compression with
    Discretized Gaussian Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun, Masaru
    Takeuchi, Jiro Katto.

    Uses residual blocks with small convolutions (3x3 and 1x1), and sub-pixel
    convolutions for up-sampling.

    Args:
        N (int): Number of channels
    """

    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, M=N, **kwargs)

        # self.g_a = nn.Sequential(
        #     ResidualBlockWithStride(3, N, stride=2),
        #     ResidualBlock(N, N),
        #     ResidualBlockWithStride(N, N, stride=2),
        #     ResidualBlock(N, N),
        #     ResidualBlockWithStride(N, N, stride=2),
        #     ResidualBlock(N, N),
        #     conv3x3(N, N, stride=2),
        # )

        self.h_a = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N, stride=2),
        )

        self.h_s = nn.Sequential(
            conv3x3(N, N),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N, N, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N, N * 3 // 2),
            nn.LeakyReLU(inplace=True),
            subpel_conv3x3(N * 3 // 2, N * 3 // 2, 2),
            nn.LeakyReLU(inplace=True),
            conv3x3(N * 3 // 2, N * 2),
        )

        # self.g_s = nn.Sequential(
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     ResidualBlock(N, N),
        #     ResidualBlockUpsample(N, N, 2),
        #     ResidualBlock(N, N),
        #     subpel_conv3x3(N, 3, 2),
        # )

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


class Cheng2020Attention_D(Cheng2020Anchor_D):
    def __init__(self, N=192, **kwargs):
        super().__init__(N=N, **kwargs)

        self.g_a_rbs1 = ResidualBlockWithStride(1, N, stride=2)
        self.g_a_rb1 = ResidualBlock(N, N)
        self.g_a_rbs2 = ResidualBlockWithStride(2 * N, N, stride=2)
        self.g_a_att1 = AttentionBlock(N)
        self.g_a_rb2 = ResidualBlock(N, N)
        self.g_a_rbs3 = ResidualBlockWithStride(2 * N, N, stride=2)
        self.g_a_rb3 = ResidualBlock(N, N)
        self.g_a_conv = conv3x3(2 * N, N, stride=2)
        self.g_a_att2 = AttentionBlock(N)

        self.g_s_att1 = AttentionBlock(N)
        self.g_s_rb1 = ResidualBlock(N, N)
        self.g_s_rbs1 = ResidualBlockUpsample(N, N, 2)
        self.g_s_rb2 = ResidualBlock(2 * N, N)
        self.g_s_rbs2 = ResidualBlockUpsample(N, N, 2)
        self.g_s_att2 = AttentionBlock(N)
        self.g_s_rb3 = ResidualBlock(2 * N, N)
        self.g_s_rbs3 = ResidualBlockUpsample(N, N, 2)
        self.g_s_rb4 = ResidualBlock(2 * N, N)
        self.g_s_conv = subpel_conv3x3(N, 1, 2)

        self.tran_conv1 = conv(2 * N, N, stride=1)
        self.tran_conv2 = conv(2 * N, N, stride=1)
        self.tran_conv3 = conv(2 * N, N, stride=1)
        self.tran_conv4 = conv(2 * N, N, stride=1)
        self.tran_conv5 = conv(2 * N, N, stride=1)
        self.tran_conv6 = conv(2 * N, N, stride=1)

        self.attention1 = ESA(N)
        self.attention2 = ESA(N)
        self.attention3 = ESA(N)
        self.attention4 = ESA(N)
        self.attention5 = ESA(N)
        self.attention6 = ESA(N)

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

    # import torchsnooper
    # @torchsnooper.snoop()
    def forward(self,x,hidden): #
        # print(type(input),type(x),type(hidden))
        g_a_g1 = hidden["ga1"]
        g_a_g2 = hidden["ga2"]
        g_a_g3 = hidden["ga3"]
        g_s_g1 = hidden["gs1"]
        g_s_g2 = hidden["gs2"]
        g_s_g3 = hidden["gs3"]

        x = self.g_a_rbs1(x)
        x = self.g_a_rb1(x)

        eg1 = self.eg_ext1(x)
        g_a_g1 = self.eg_ext2(g_a_g1)
        g_a_g1 = self.tran_conv1(torch.cat((eg1, g_a_g1), dim=-3))
        g_a_g1 = self.attention1(g_a_g1)

        x = self.g_a_rbs2(torch.cat((x, g_a_g1), dim=-3))
        x = self.g_a_att1(x)
        x = self.g_a_rb2(x)

        eg2 = self.eg_ext3(x)
        g_a_g2 = self.eg_ext4(g_a_g2)
        g_a_g2 = self.tran_conv2(torch.cat((eg2, g_a_g2), dim=-3))
        g_a_g2 = self.attention2(g_a_g2)

        x = self.g_a_rbs3(torch.cat((x, g_a_g2), dim=-3))
        x = self.g_a_rb3(x)

        eg3 = self.eg_ext5(x)
        g_a_g3 = self.eg_ext6(g_a_g3)
        g_a_g3 = self.tran_conv3(torch.cat((eg3, g_a_g3), dim=-3))
        g_a_g3 = self.attention3(g_a_g3)

        x = self.g_a_conv(torch.cat((x, g_a_g3), dim=-3))
        y2 = self.g_a_att2(x)

        z2 = self.h_a(y2)
        z2_hat, z2_likelihoods = self.entropy_bottleneck(z2)
        params = self.h_s(z2_hat)

        y2_hat = self.gaussian_conditional.quantize(
            y2, "noise" if self.training else "dequantize"
        )
        ctx_params = self.context_prediction(y2_hat)
        gaussian_params = self.entropy_parameters(
            torch.cat((params, ctx_params), dim=1)
        )
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        _, y2_likelihoods = self.gaussian_conditional(y2, scales_hat, means=means_hat)

        x = self.g_s_att1(y2_hat)
        x = self.g_s_rb1(x)
        x = self.g_s_rbs1(x)

        eg4 = self.eg_ext7(x)
        g_s_g1 = self.eg_ext8(g_s_g1)
        g_s_g1 = self.tran_conv4(torch.cat((eg4, g_s_g1), dim=-3))
        g_s_g1 = self.attention4(g_s_g1)

        x = self.g_s_rb2(torch.cat((x, g_s_g1), dim=-3))
        x = self.g_s_rbs2(x)
        x = self.g_s_att2(x)

        eg5 = self.eg_ext9(x)
        g_s_g2 = self.eg_ext10(g_s_g2)
        g_s_g2 = self.tran_conv5(torch.cat((eg5, g_s_g2), dim=-3))
        g_s_g2 = self.attention5(g_s_g2)

        x = self.g_s_rb3(torch.cat((x, g_s_g2), dim=-3))
        x = self.g_s_rbs3(x)

        eg6 = self.eg_ext11(x)
        g_s_g3 = self.eg_ext12(g_s_g3)
        g_s_g3 = self.tran_conv6(torch.cat((eg6, g_s_g3), dim=-3))
        g_s_g3 = self.attention6(g_s_g3)

        x = self.g_s_rb4(torch.cat((x, g_s_g3), dim=-3))
        x2_hat = self.g_s_conv(x)

        return {
            "x_hat": x2_hat,
            "likelihoods": {"y": y2_likelihoods, "z": z2_likelihoods},
        }

    