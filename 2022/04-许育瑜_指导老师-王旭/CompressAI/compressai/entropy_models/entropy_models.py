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

from re import X
import warnings

from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from compressai._CXX import pmf_to_quantized_cdf as _pmf_to_quantized_cdf
from compressai.ops import LowerBound
# import torchsnooper


class _EntropyCoder:
    """Proxy class to an actual entropy coder class."""

    def __init__(self, method):
        if not isinstance(method, str):
            raise ValueError(f'Invalid method type "{type(method)}"')

        from compressai import available_entropy_coders

        if method not in available_entropy_coders():
            methods = ", ".join(available_entropy_coders())
            raise ValueError(
                f'Unknown entropy coder "{method}"' f" (available: {methods})"
            )

        if method == "ans":
            from compressai import ans

            encoder = ans.RansEncoder()
            decoder = ans.RansDecoder()
        elif method == "rangecoder":
            import range_coder

            encoder = range_coder.RangeEncoder()
            decoder = range_coder.RangeDecoder()

        self.name = method
        self._encoder = encoder
        self._decoder = decoder

    def encode_with_indexes(self, *args, **kwargs):
        return self._encoder.encode_with_indexes(*args, **kwargs)

    def decode_with_indexes(self, *args, **kwargs):
        return self._decoder.decode_with_indexes(*args, **kwargs)


def default_entropy_coder():
    from compressai import get_entropy_coder

    return get_entropy_coder()


def pmf_to_quantized_cdf(pmf: Tensor, precision: int = 16) -> Tensor:
    cdf = _pmf_to_quantized_cdf(pmf.tolist(), precision)
    cdf = torch.IntTensor(cdf)
    return cdf


def _forward(self, *args: Any) -> Any:
    raise NotImplementedError()


class EntropyModel(nn.Module):
    r"""Entropy model base class.

    Args:
        likelihood_bound (float): minimum likelihood bound
        entropy_coder (str, optional): set the entropy coder to use, use default
            one if None
        entropy_coder_precision (int): set the entropy coder precision
    """

    def __init__(
        self,
        likelihood_bound: float = 1e-9,
        entropy_coder: Optional[str] = None,
        entropy_coder_precision: int = 16,
    ):
        super().__init__()

        if entropy_coder is None:
            entropy_coder = default_entropy_coder()
        self.entropy_coder = _EntropyCoder(entropy_coder)
        self.entropy_coder_precision = int(entropy_coder_precision)

        self.use_likelihood_bound = likelihood_bound > 0  # true
        if self.use_likelihood_bound:
            self.likelihood_lower_bound = LowerBound(likelihood_bound)  #  self.likelihood_lower_bound是一个对象 1e-9    # 会被guassEntroy继承

        # to be filled on update()
        self.register_buffer("_offset", torch.IntTensor())         # 这里注册了offset，quantized_cdf,cdf_length，buffer不是参数不参与网络的更新，但这几个会在update函数，更新完给编码器使用
        self.register_buffer("_quantized_cdf", torch.IntTensor())
        self.register_buffer("_cdf_length", torch.IntTensor())

    def __getstate__(self):
        attributes = self.__dict__.copy()
        attributes["entropy_coder"] = self.entropy_coder.name
        return attributes

    def __setstate__(self, state):
        self.__dict__ = state
        self.entropy_coder = _EntropyCoder(self.__dict__.pop("entropy_coder"))

    @property
    def offset(self):
        return self._offset

    @property
    def quantized_cdf(self):
        return self._quantized_cdf

    @property
    def cdf_length(self):
        return self._cdf_length

    # See: https://github.com/python/mypy/issues/8795
    forward: Callable[..., Any] = _forward

    def quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        if mode not in ("noise", "dequantize", "symbols"):
            raise ValueError(f'Invalid quantization mode: "{mode}"')

        if mode == "noise":
            half = float(0.5)
            noise = torch.empty_like(inputs).uniform_(-half, half)
            inputs = inputs + noise
            return inputs

        outputs = inputs.clone()
        if means is not None:
            outputs -= means

        outputs = torch.round(outputs)

        if mode == "dequantize":
            if means is not None:
                outputs += means
            return outputs

        assert mode == "symbols", mode
        outputs = outputs.int()
        return outputs

    def _quantize(
        self, inputs: Tensor, mode: str, means: Optional[Tensor] = None
    ) -> Tensor:
        warnings.warn("_quantize is deprecated. Use quantize instead.")
        return self.quantize(inputs, mode, means)

    @staticmethod
    def dequantize(
        inputs: Tensor, means: Optional[Tensor] = None, dtype: torch.dtype = torch.float
    ) -> Tensor:
        if means is not None:
            outputs = inputs.type_as(means)
            outputs += means
        else:
            outputs = inputs.type(dtype)
        return outputs

    @classmethod
    def _dequantize(cls, inputs: Tensor, means: Optional[Tensor] = None) -> Tensor:
        warnings.warn("_dequantize. Use dequantize instead.")
        return cls.dequantize(inputs, means)

    def _pmf_to_cdf(self, pmf, tail_mass, pmf_length, max_length):
        cdf = torch.zeros(
            (len(pmf_length), max_length + 2), dtype=torch.int32, device=pmf.device
        )
        for i, p in enumerate(pmf):
            prob = torch.cat((p[: pmf_length[i]], tail_mass[i]), dim=0)
            _cdf = pmf_to_quantized_cdf(prob, self.entropy_coder_precision)
            cdf[i, : _cdf.size(0)] = _cdf
        return cdf

    def _check_cdf_size(self):
        if self._quantized_cdf.numel() == 0:
            raise ValueError("Uninitialized CDFs. Run update() first")

        if len(self._quantized_cdf.size()) != 2:
            raise ValueError(f"Invalid CDF size {self._quantized_cdf.size()}")

    def _check_offsets_size(self):
        if self._offset.numel() == 0:
            raise ValueError("Uninitialized offsets. Run update() first")

        if len(self._offset.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._offset.size()}")

    def _check_cdf_length(self):
        if self._cdf_length.numel() == 0:
            raise ValueError("Uninitialized CDF lengths. Run update() first")

        if len(self._cdf_length.size()) != 1:
            raise ValueError(f"Invalid offsets size {self._cdf_length.size()}")

    def compress(self, inputs, indexes, means=None):
        """
        Compress input tensors to char strings.

        Args:
            inputs (torch.Tensor): input tensors
            indexes (torch.IntTensor): tensors CDF indexes
            means (torch.Tensor, optional): optional tensor means
        """
        symbols = self.quantize(inputs, "symbols", means)

        if len(inputs.size()) < 2:
            raise ValueError(
                "Invalid `inputs` size. Expected a tensor with at least 2 dimensions."
            )

        if inputs.size() != indexes.size():  # inputs和indexes具有相同的size
            raise ValueError("`inputs` and `indexes` should have the same size.")

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        strings = []
        for i in range(symbols.size(0)):                 # 实际上是batch值，这里每次处理一张图片
            rv = self.entropy_coder.encode_with_indexes(
                symbols[i].reshape(-1).int().tolist(),   # 64*8*8 且转换成int值，转换成list，方便c++处理的时候，转换成vector
                indexes[i].reshape(-1).int().tolist(),   # 64*8*8 
                self._quantized_cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            strings.append(rv)
        return strings

    def decompress(
        self,
        strings: str,
        indexes: torch.IntTensor,
        dtype: torch.dtype = torch.float,
        means: torch.Tensor = None,
    ):
        """
        Decompress char strings to tensors.

        Args:
            strings (str): compressed tensors
            indexes (torch.IntTensor): tensors CDF indexes
            dtype (torch.dtype): type of dequantized output
            means (torch.Tensor, optional): optional tensor means
        """

        if not isinstance(strings, (tuple, list)):
            raise ValueError("Invalid `strings` parameter type.")

        if not len(strings) == indexes.size(0):
            raise ValueError("Invalid strings or indexes parameters")

        if len(indexes.size()) < 2:
            raise ValueError(
                "Invalid `indexes` size. Expected a tensor with at least 2 dimensions."
            )

        self._check_cdf_size()
        self._check_cdf_length()
        self._check_offsets_size()

        if means is not None:
            if means.size()[:2] != indexes.size()[:2]:
                raise ValueError("Invalid means or indexes parameters")
            if means.size() != indexes.size():
                for i in range(2, len(indexes.size())):
                    if means.size(i) != 1:
                        raise ValueError("Invalid means parameters")

        cdf = self._quantized_cdf
        outputs = cdf.new_empty(indexes.size())

        for i, s in enumerate(strings):
            values = self.entropy_coder.decode_with_indexes(
                s,
                indexes[i].reshape(-1).int().tolist(),
                cdf.tolist(),
                self._cdf_length.reshape(-1).int().tolist(),
                self._offset.reshape(-1).int().tolist(),
            )
            outputs[i] = torch.tensor(
                values, device=outputs.device, dtype=outputs.dtype
            ).reshape(outputs[i].size())
        outputs = self.dequantize(outputs, means, dtype)
        return outputs


class EntropyBottleneck(EntropyModel):
    r"""Entropy bottleneck layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the entropy bottleneck layer in
    *tensorflow/compression*. See the original paper and the `tensorflow
    documentation
    <https://tensorflow.github.io/compression/docs/entropy_bottleneck.html>`__
    for an introduction.
    """

    _offset: Tensor

    def __init__(
        self,
        channels: int,
        *args: Any,
        tail_mass: float = 1e-9,
        init_scale: float = 10,
        filters: Tuple[int, ...] = (3, 3, 3, 3),
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        self.channels = int(channels)                               # 64
        self.filters = tuple(int(f) for f in filters)               # (3,3,3,3)
        self.init_scale = float(init_scale)                         # 10.0
        self.tail_mass = float(tail_mass)                           # 1e-9

        # Create parameters
        filters = (1,) + self.filters + (1,)                        # (1,3,3,3,3,1)
        scale = self.init_scale ** (1 / (len(self.filters) + 1))    # 10^(1/5)
        channels = self.channels                                    # 64

        # 对H、b、a进行初始化
        for i in range(len(self.filters) + 1):                      # range(5)
            init = np.log(np.expm1(1 / scale / filters[i + 1]))     # log(e^(1/(10^(1/5)/?))-1) ?=(3,3,3,3,1)  【问题：】为什么初始化这么复杂？（初始化问题，反而不太好理解，是随机的，还是经过实验的，还是有数学依据的？）
            matrix = torch.Tensor(channels, filters[i + 1], filters[i])     # (64,?,!) ?=(3,3,3,3,1) !=(1,3,3,3,3) 为了方便乘法
            matrix.data.fill_(init)
            self.register_parameter(f"_matrix{i:d}", nn.Parameter(matrix))

            bias = torch.Tensor(channels, filters[i + 1], 1)
            nn.init.uniform_(bias, -0.5, 0.5)
            self.register_parameter(f"_bias{i:d}", nn.Parameter(bias))

            if i < len(self.filters):
                factor = torch.Tensor(channels, filters[i + 1], 1)
                nn.init.zeros_(factor)
                self.register_parameter(f"_factor{i:d}", nn.Parameter(factor))

        self.quantiles = nn.Parameter(torch.Tensor(channels, 1, 3))  # [64,1,3]
        init = torch.Tensor([-self.init_scale, 0, self.init_scale])  # (-10,0,10)
        self.quantiles.data = init.repeat(self.quantiles.size(0), 1, 1) # 64*(-10,0,10) 
 
        target = np.log(2 / self.tail_mass - 1)                      # log(2/1e-9 -1)
        self.register_buffer("target", torch.Tensor([-target, 0, target])) # (log(2/1e-9 -1),0,log(2/1e-9 -1))

    def _get_medians(self) -> Tensor:
        medians = self.quantiles[:, :, 1:2]   # [64,1,1]  如果有使用冒号，那么维数不变
        return medians

    
    
    # 更新self.offset\cdf\cdf_length给编码器使用
    # @torchsnooper.snoop()
    def update(self, force: bool = False) -> bool:
        # Check if we need to update the bottleneck parameters, the offsets are
        # only computed and stored when the conditonal model is update()'d.
        if self._offset.numel() > 0 and not force:  # numel获取张量中元素的个数
            return False

        medians = self.quantiles[:, 0, 1]            # [64,1,3] [64] 如果没有使用冒号，那么会进行降维

        minima = medians - self.quantiles[:, 0, 0]   # [64]  每个维度共用一个分布，所以，需要估计64个量级
        minima = torch.ceil(minima).int()            # 本来是float类型，后面需要参与symbol（整数）的计算
        minima = torch.clamp(minima, min=0)          

        maxima = self.quantiles[:, 0, 2] - medians   
        maxima = torch.ceil(maxima).int()
        maxima = torch.clamp(maxima, min=0)

        self._offset = -minima                       # 向左偏移量，可能为负数   [64]           # 这里更新，是给后面真正编码的时候使用的数值

        pmf_start = medians - minima                 # 向左偏移，作为初始地址   [64]           # medians - ceil(medians - left) ≈ left 都是小数？因为medians是小数
        # feature的值比较大，那么中间的整数会比较多，所以pmf会比较长;如果feature的值比较小，那么中间就没什么整数了，所以pmf会比较短？
        pmf_length = maxima + minima + 1             # 整体长度  [64]                           # 整数

        max_length = pmf_length.max().item()         # 最大能够存放的长度       max([64]),假设是32？
        device = pmf_start.device

        # 通过最大范围单位，构建出符合feature分布的sample，然后再进行计算。这样做的好处体现：小范围的通道，并不需要使用很多的pmf和cdf值，这就相比于forward简化了大量的计算。
        samples = torch.arange(max_length, device=device)       # range(32) 把最大通道的范围分配下来,有点相当于标准坐标系，只有正值的标准坐标系
        samples = samples[None, :] + pmf_start[:, None, None]   # 广播加法，samples.shape = [64,1,32]  # 向左偏移，形成与轴对称的坐标系 

        half = float(0.5)

        # 【问题：】如果是之前的feature，为什么这里的sample可以充当feature？
        # 计算samples对应的概率
        lower = self._logits_cumulative(samples - half, stop_gradient=True)             # 这里stop_gradient=True，说明了H，b，a没有参与更新
        upper = self._logits_cumulative(samples + half, stop_gradient=True)             # [64,1,32] 
        sign = -torch.sign(lower + upper)
        pmf = torch.abs(torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower))      # [64,1,32]

        pmf = pmf[:, 0, :]                                                              # [64,32] 理论上来说，每个像素，都会对应一个概率
        tail_mass = torch.sigmoid(lower[:, 0, :1]) + torch.sigmoid(-upper[:, 0, -1:])   # :00 + :0-1

        # 分别对每一行的pmf加上tail_mass,然后转化成量化的_cdfs，然后再装载进cdf中
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)        # [64,32+2] 范围：0-65536
        self._quantized_cdf = quantized_cdf  # 这是在真正编码的时候，才使用到的cdf，否则的话，通过forward，直接预测pmf，然后估计bit就行了
        self._cdf_length = pmf_length + 2
        return True

    # 辅助损失：使得quantiles==》[-x,0,x],其中c(-x)=0,c(x)=1?  也就是说，这个函数是用来限制quantiles的（因为quantiles是parameter，而target是buffer）
    # quantile使用forward更新好的H a b参数，进行了一系列的线性变换和非线性变换之后，能够接近(-21,0,21)  
    # 限制住quantile，那么相当于H a b 都是有效的。
    # 是有两个不同的optimizer，是没有联合训练的。看起来forward的过程，才是论文中真正的推导过程，但是为什么又多了一个辅助loss呢？
    # 用来估计feature的量级，只有当quantiles和feature处在同一个量级的时候，H a b 才有效。
    # 但是想要更好地实际理解编解码的过程的话，就必须理解loss的含义，以及quantile的作用。
    # @torchsnooper.snoop()
    def loss(self) -> Tensor:  
        # print(self.quantiles[:10])
        logits = self._logits_cumulative(self.quantiles, stop_gradient=True)   # 计算f_(k-1)(quantiles) ==>target [64,1,3]  # 注意这里也是没有更新H a b参数的
        loss = torch.abs(logits - self.target).sum()                           # |logits-[-21,0,21]|.sum  广播机制，从右向左匹配，要么维度为1，要么没有维度
        return loss                                                           
    
    # 返回最后fk=sigmoid(x)的x
    def _logits_cumulative(self, inputs: Tensor, stop_gradient: bool) -> Tensor:
        # TorchScript not yet working (nn.Mmodule indexing not supported)
        logits = inputs
        # 迭代：
        for i in range(len(self.filters) + 1):  # filters=[3,3,3,3],range(5)  由5个函数复合而成
            matrix = getattr(self, f"_matrix{i:d}")
            if stop_gradient:
                matrix = matrix.detach()
            logits = torch.matmul(F.softplus(matrix), logits)  

            bias = getattr(self, f"_bias{i:d}")
            if stop_gradient:
                bias = bias.detach()
            logits += bias  # x = Hx+b                                        

            if i < len(self.filters):          # i<4  说明了最后只是输出了Hx+b  [H>0,其他没有限制]
                factor = getattr(self, f"_factor{i:d}")
                if stop_gradient:
                    factor = factor.detach()
                logits += torch.tanh(factor) * torch.tanh(logits)  # g(x) = x + tanh(a) * tanh(x) in [0,1]
        return logits  

    # 作用：先计算出Hx+b，然后计算出cdf，然后再去计算pmf
    @torch.jit.unused  # 允许您在模型中保留与 TorchScript 不兼容的代码，但仍可以导出模型
    def _likelihood(self, inputs: Tensor) -> Tensor:
        half = float(0.5)
        v0 = inputs - half
        v1 = inputs + half
        lower = self._logits_cumulative(v0, stop_gradient=False)  # 没有\mu,\sigma
        upper = self._logits_cumulative(v1, stop_gradient=False)
        sign = -torch.sign(lower + upper)  # 假设lower=-0.3 upper=0.5,那么sign=-
        sign = sign.detach()
        likelihood = torch.abs(
            torch.sigmoid(sign * upper) - torch.sigmoid(sign * lower)  # |S(x)-S(y)|=|S(-x)-S(-y)| 通过sigmoid公式可以证明，但是为什么要有sign符号呢？
        )
        return likelihood

    # 量化和计算概率
    def forward(
        self, x: Tensor, training: Optional[bool] = None
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training

        if not torch.jit.is_scripting():   # 编译时，返回True
            # x from B x C x ... to C x B x ...
            perm = np.arange(len(x.shape))       # range(4)
            perm[0], perm[1] = perm[1], perm[0]  # [1,0,2,3]
            # Compute inverse permutation
            inv_perm = np.arange(len(x.shape))[np.argsort(perm)]   # 从小到大的参数  [0,1,2,3][1,0,2,3]=[1,0,2,3] 
        else:
            raise NotImplementedError()
            # TorchScript in 2D for static inference
            # Convert to (channels, ... , batch) format
            # perm = (1, 2, 3, 0)
            # inv_perm = (3, 0, 1, 2)

        x = x.permute(*perm).contiguous()
        shape = x.size()                        # [64,16,8,8]
        values = x.reshape(x.size(0), 1, -1)    # [64,1,16*8*8]  把batch中也集中在一起了？

        # Add noise or quantize

        outputs = self.quantize(
            values, "noise" if training else "dequantize", self._get_medians()
        )

        if not torch.jit.is_scripting():
            likelihood = self._likelihood(outputs)
            if self.use_likelihood_bound:
                likelihood = self.likelihood_lower_bound(likelihood)
        else:
            raise NotImplementedError()
            # TorchScript not yet supported
            # likelihood = torch.zeros_like(outputs)

        # Convert back to input tensor shape 恢复成 [16,64,8,8]
        outputs = outputs.reshape(shape)
        outputs = outputs.permute(*inv_perm).contiguous()

        likelihood = likelihood.reshape(shape)
        likelihood = likelihood.permute(*inv_perm).contiguous()

        return outputs, likelihood

    @staticmethod
    def _build_indexes(size):  # nchw  假设16*64*8*8   # 既然是一一对应的关系，那么为什么要采取这样复杂的扩充方式？是为了以后的扩展性吗？
        dims = len(size)       # 4
        N = size[0]                     
        C = size[1]

        view_dims = np.ones((dims,), dtype=np.int64)  # (4,)
        view_dims[1] = -1                             # (1,-1,1,1)                      
        indexes = torch.arange(C).view(*view_dims)    # arange是没有包括最后一个数的，生成一个向量    range(0,64).view(1,-1,1,1)
        indexes = indexes.int()

        return indexes.repeat(N, 1, *size[2:])        # batch中每张图片上的每个像素，都记录自己对应着第index个通道

    @staticmethod
    def _extend_ndims(tensor, n):
        return tensor.reshape(-1, *([1] * n)) if n > 0 else tensor.reshape(-1)

    def compress(self, x):                        # x为encoder提取的特征 [16,64,8,8]
        indexes = self._build_indexes(x.size())   # 记录特征的每个的像素对应着的通道 [把特征也看成图片]
        medians = self._get_medians().detach()    # [64,1,1] 
        spatial_dims = len(x.size()) - 2          # 2
        medians = self._extend_ndims(medians, spatial_dims)  # 扩充两个维度，可能是针对medians如果是1维的情况，扩充到3维
        medians = medians.expand(x.size(0), *([-1] * (spatial_dims + 1)))  # expand(16,-1,-1,-1)=[16,64,1,1] 将形状广播成expand的参数的形状
        # 实际上还传入了self._quantized_cdf.tolist(),self._cdf_length.reshape(-1).int().tolist(),self._offset.reshape(-1).int().tolist(),
        # 注意：这里的x有经过处理成分别每张图片，并把特征都转换成1维，然后再传入编码其中压缩
        return super().compress(x, indexes, medians)                       # x 

    def decompress(self, strings, size):
        output_size = (len(strings), self._quantized_cdf.size(0), *size)
        indexes = self._build_indexes(output_size).to(self._quantized_cdf.device)
        medians = self._extend_ndims(self._get_medians().detach(), len(size))
        medians = medians.expand(len(strings), *([-1] * (len(size) + 1)))
        return super().decompress(strings, indexes, medians.dtype, medians)


class GaussianConditional(EntropyModel):
    r"""Gaussian conditional layer, introduced by J. Ballé, D. Minnen, S. Singh,
    S. J. Hwang, N. Johnston, in `"Variational image compression with a scale
    hyperprior" <https://arxiv.org/abs/1802.01436>`_.

    This is a re-implementation of the Gaussian conditional layer in
    *tensorflow/compression*. See the `tensorflow documentation
    <https://tensorflow.github.io/compression/docs/api_docs/python/tfc/GaussianConditional.html>`__
    for more information.
    """

    def __init__(
        self,
        scale_table: Optional[Union[List, Tuple]],
        *args: Any,
        scale_bound: float = 0.11,
        tail_mass: float = 1e-9,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)

        if not isinstance(scale_table, (type(None), list, tuple)):
            raise ValueError(f'Invalid type for scale_table "{type(scale_table)}"')

        if isinstance(scale_table, (list, tuple)) and len(scale_table) < 1:
            raise ValueError(f'Invalid scale_table length "{len(scale_table)}"')

        if scale_table and ( scale_table != sorted(scale_table) or any(s <= 0 for s in scale_table)):
            raise ValueError(f'Invalid scale_table "({scale_table})"')

        # scale_table=None
        self.tail_mass = float(tail_mass)
        if scale_bound is None and scale_table:
            scale_bound = self.scale_table[0]
        if scale_bound <= 0:
            raise ValueError("Invalid parameters")
        self.lower_bound_scale = LowerBound(scale_bound)  # \sigma_min=0.11

        self.register_buffer(
            "scale_table",
            self._prepare_scale_table(scale_table) if scale_table else torch.Tensor(),
        )

        self.register_buffer(
            "scale_bound",
            torch.Tensor([float(scale_bound)]) if scale_bound is not None else None,
        )

    @staticmethod
    def _prepare_scale_table(scale_table):
        return torch.Tensor(tuple(float(s) for s in scale_table))

    def _standardized_cumulative(self, inputs: Tensor) -> Tensor:
        half = float(0.5)           # 0.5
        const = float(-(2**-0.5))   # -0.7071   -sqrt(2)/2
        # Using the complementary error function maximizes numerical precision.
        # erfc(x)=1-erf(x)
        # C(x)=0.5*(1+erf(x/sqrt(2)))=0.5*(1+1-erfc(x/sqrt(2)))=0.5(2-2erfc(x/sqrt(2)))
        return half * torch.erfc(const * inputs)
        

    @staticmethod
    def _standardized_quantile(quantile):
        return scipy.stats.norm.ppf(quantile)  # ppf是cdf的逆函数，也就是给定cdf，求x的值

    # 更新scale_table,同时更新模型参数
    def update_scale_table(self, scale_table, force=False):
        # Check if we need to update the gaussian conditional parameters, the
        # offsets are only computed and stored when the conditonal model is
        # updated.
        if self._offset.numel() > 0 and not force:
            return False
        device = self.scale_table.device
        self.scale_table = self._prepare_scale_table(scale_table).to(device)  # 转化成float的tuple
        self.update()                                                         # 调用update_scale_table的时候，已经调用了update
        return True

    # 更新cdf、offset、cdf_length
    def update(self):
        multiplier = -self._standardized_quantile(self.tail_mass / 2)  # 6.109 默认是标准正态分布，尾数==》逆映射出两边的边界 N(0,1)
        pmf_center = torch.ceil(self.scale_table * multiplier).int()   # (x-\mu/\sigma) 因为默认mean=0，这相当于给定每个scale_table对应的边界，所以命名center..
        pmf_length = 2 * pmf_center + 1                                
        max_length = torch.max(pmf_length).item()                      # (256*6)*2+1=3073  *64个scale 所以这个sample还是蛮大的

        device = pmf_center.device
        samples = torch.abs(
            torch.arange(max_length, device=device).int() - pmf_center[:, None]  # 向左偏移，形成中心对称的坐标系
        )
        samples_scale = self.scale_table.unsqueeze(1)
        samples = samples.float()
        samples_scale = samples_scale.float()
        upper = self._standardized_cumulative((0.5 - samples) / samples_scale)
        lower = self._standardized_cumulative((-0.5 - samples) / samples_scale)
        pmf = upper - lower

        tail_mass = 2 * lower[:, :1]

        quantized_cdf = torch.Tensor(len(pmf_length), max_length + 2)
        quantized_cdf = self._pmf_to_cdf(pmf, tail_mass, pmf_length, max_length)
        self._quantized_cdf = quantized_cdf
        self._offset = -pmf_center
        self._cdf_length = pmf_length + 2

        vardict = locals()        
        def myprint(var,vardict):
            varname = [k for k,v in vardict.items() if v is var][0]
            # var = var.cpu().detach().numpy()
            # np.save(f'var_{varname}.npy',var)
            print(varname,var.shape)
            return True
        print('cdf max length',max_length)
        printlist = [pmf_center,pmf_length,samples,quantized_cdf]
        [myprint(x,vardict) for x in printlist]

    # 因为input符合正态分布，input=x==》cdf（x）==》pmf（x），这个映射过程是一一对应的
    def _likelihood(
        self, inputs: Tensor, scales: Tensor, means: Optional[Tensor] = None
    ) -> Tensor:
        half = float(0.5)

        if means is not None:
            values = inputs - means  # N(\mu,\sigma^2) or N(0,\sigma^2)
        else:
            values = inputs
       
        scales = self.lower_bound_scale(scales)  # max(0.11,\sigma)

        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)    # C(0.5-val)  为什么使用-val？  # 因为y的均值为0，且符合正态分布，因此y跟-y都可？
        lower = self._standardized_cumulative((-half - values) / scales)   # C(-0.5-val)
        likelihood = upper - lower                                         # p=C(y+0.5)-C(y-0.5) 具体参考balle论文

        return likelihood


    # @torchsnooper.snoop()
    # 作用：根据特征值返回概率
    # 返回值：量化后的input，以及input每个像素对应的概率
    def forward(
        self,
        inputs: Tensor,
        scales: Tensor,
        means: Optional[Tensor] = None,
        training: Optional[bool] = None,
    ) -> Tuple[Tensor, Tensor]:
        if training is None:
            training = self.training
        

        outputs = self.quantize(inputs, "noise" if training else "dequantize", means)   # 将encoder提取的特征进行量化

        likelihood = self._likelihood(outputs, scales, means)                           # 计算出概率
        if self.use_likelihood_bound:
            likelihood = self.likelihood_lower_bound(likelihood)                        # 保证概率都大于1e-9  问题：这里的概率，是指每个特征值对应的概率吗？
        return outputs, likelihood

    # scales相当于传入\sigma  
    # 这是在编码的时候传入的，而不是在update的时候传入的，有什么区别呢?
    def build_indexes(self, scales: Tensor) -> Tensor:  # 每个像素点，应该对应一个索引：[16,64,8,8]
        scales = self.lower_bound_scale(scales)         # max(0.11,\sigma)  # 为什么要限制bound？只是一个随意的超参数【实际上前面写死，这里还需要lower_bound,有时为了灵活性？】
        indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()   # [16,64,8,8] 全部初始化成63；类似于ones操作，但是可以保证device和dtype跟之前的变量一样
        for s in self.scale_table[:-1]:                                             # 为什么舍弃最后一个数值？初始化的时候是63，最多减63次，需要保证index>=0
            indexes -= (scales <= s).int()                                          # s in [0.11,256] 有的大于63，有的小于63，
        return indexes                                                              # 其中scales是不会变化的，s在不断变大，而index则在不断变小【换种表达方式：scales小的，对应index小的】
