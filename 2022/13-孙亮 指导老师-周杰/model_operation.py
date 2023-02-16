import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
import numpy as np


OPS = {
    'identity': lambda C_in, stride, affine: Identity() if stride == 1 else FactorizedReduce(C_in, C_in, affine=affine),
    'max_pool_3x3': lambda C_in, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
    'avg_pool_3x3': lambda C_in, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'squeeze_and_excitation': lambda C_in, stride, affine : SqueezeExcitation(C_in, stride=stride),
    'lb_conv_3x3': lambda C_in, stride, affine: LBConv(C_in, C_in, 3, stride, padding=1),
    'lb_conv_5x5': lambda C_in, stride, affine: LBConv(C_in, C_in, 5, stride, padding=2),
    'dil_conv_3x3': lambda C_in, stride, affine: DilConv(C_in, C_in, 3, stride, 2, 2, affine=affine),
    'dil_conv_5x5': lambda C_in, stride, affine: DilConv(C_in, C_in, 5, stride, 4, 2, affine=affine),
    'sep_conv_3x3': lambda C_in, stride, affine: SepConv(C_in, C_in, 3, stride, 1, affine=affine),
    'sep_conv_5x5': lambda C_in, stride, affine: SepConv(C_in, C_in, 5, stride, 2, affine=affine),
    'sep_conv_7x7': lambda C_in, stride, affine: SepConv(C_in, C_in, 7, stride, 3, affine=affine),
    'conv_7x1_1x7': lambda C_in, stride, affine: Con7x1_1x7(C_in, C_in, stride, affine=affine),
}


def _make_divisible(ch, divisor=8, min_ch=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


class Identity(nn.Module):

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class SqueezeExcitation(nn.Module):

    def __init__(self, input_c : int, squeeze_factor: int = 4, stride: int = 1):
        super(SqueezeExcitation, self).__init__()
        self.stride = stride
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        if stride != 1:
            self.half_size = FactorizedReduce(input_c, input_c)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        if self.stride != 1:
            x = self.half_size(x)
        scale = F.adaptive_avg_pool2d(x, output_size=(1,1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class DilConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
        super(DilConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                      groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class SepConv(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(SepConv, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_in, affine=affine),
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
            nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(C_out, affine=affine),
        )

    def forward(self, x):
        return self.op(x)


class Con7x1_1x7(nn.Module):

    def __init__(self, C_in, C_out, stride, affine=True):
        super(Con7x1_1x7, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_in, (1, 7), stride=(1,stride), padding=(0, 3), bias=False),
            nn.Conv2d(C_in, C_out, (7, 1), stride=(stride, 1), padding=(3, 0), bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


# Local Binary Convolution
class RandomBinaryConv(nn.Module):
    """Random Binary Convolution.

    See Local Binary Convolutional Neural Networks.
    from:
    https://github.com/juefeix/lbcnn.pytorch/blob/main/lbcnn/models.py
    """

    def __init__(self, in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 sparsity=0.9,
                 bias=False,
                 padding=0,
                 groups=1,
                 dilation=1,
                 seed=1234):
        """
        TODO(zcq) Write a cuda/c++ version.
        Parameters
        ----------
        sparsity : float
        """
        super().__init__()
        self.out_channels = out_channels
        self.in_channels = in_channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        num_elements = out_channels * in_channels * kernel_size * kernel_size
        assert not bias, "bias=True not supported"
        weight = torch.zeros((num_elements,), requires_grad=False).float()
        index = np.random.choice(num_elements, int(sparsity * num_elements))
        weight[index] = torch.bernoulli(torch.ones_like(weight)[index] * 0.5) * 2 - 1
        weight = weight.view((out_channels, in_channels, kernel_size, kernel_size))
        self.register_buffer('weight', weight)

    def forward(self, x):
        return F.conv2d(x, self.weight, stride=self.stride,
                        padding=self.padding, dilation=self.dilation,
                        groups=self.groups)


class LBConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 sparsity=0.9,
                 bias=False,
                 seed=1234,
                 act=F.relu):
        """Use this to replace a conv + activation.
        """
        super().__init__()
        self.random_binary_conv = RandomBinaryConv(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            sparsity=sparsity,
            seed=seed)
        # self.bn = nn.BatchNorm2d(in_channels)
        self.fc = nn.Conv2d(out_channels, out_channels, 1, 1)
        self.act = act

    def forward(self, x):
        # y = self.bn(x)
        y = self.random_binary_conv(x)
        if self.act is not None:
            y = self.act(y)
        y = self.fc(y)
        return y


class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
        super(ReLUConvBN, self).__init__()
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(C_out, affine=affine)
        )

    def forward(self, x):
        return self.op(x)


class FactorizedReduce(nn.Module):

    def __init__(self, C_in, C_out, affine=True):
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.relu = nn.ReLU(inplace=False)
        self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(C_out, affine=affine)

    def forward(self, x):
        x = self.relu(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.bn(out)
        return out

