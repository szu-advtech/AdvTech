import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image


def GaussProjection(x, mean, std):
    scale = math.sqrt(2 * math.pi) * std
    x_out = torch.exp(-(x - mean) ** 2 / (2 * std ** 2))
    x_out = x_out / scale
    return x_out


class ChannelAMM(nn.Module):
    def __init__(self, in_features, reduction=16):
        super(ChannelAMM, self).__init__()
        self.conv = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, in_features // reduction),
            nn.ReLU(),
            nn.Linear(in_features // reduction, in_features)
        )

    def forward(self, x):
        x_out = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        x_out = self.conv(x_out)
        mean = torch.mean(x_out).detach()
        std = torch.std(x_out).detach()
        return x * GaussProjection(x_out, mean, std).unsqueeze(2).unsqueeze(3).expand_as(x)


class ChannelPool(nn.Module):
    def forward(self, x):
        # TODO 尝试加max和不加max，对比两种操作的结果
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialAMM(nn.Module):
    def __init__(self):
        super(SpatialAMM, self).__init__()
        kernel_size = 7
        self.avg_pool = ChannelPool()
        self.conv = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        x_out = self.avg_pool(x)
        x_out = self.conv(x_out)
        mean = torch.mean(x_out).detach()
        std = torch.std(x_out).detach()
        return x * GaussProjection(x_out, mean, std)


class AMM(nn.Module):
    def __init__(self, in_features, reduction):
        super(AMM, self).__init__()
        self.ChannelAMM = ChannelAMM(in_features=in_features, reduction=reduction)
        self.SpatialAMM = SpatialAMM()

    def forward(self, x):
        x_out = self.ChannelAMM(x)
        x_out = self.SpatialAMM(x_out)
        return x_out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
