import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import (MetaModule, MetaConv2d, MetaBatchNorm2d, MetaLinear)
from typing import Dict, Optional
from collections import OrderedDict
#from torchmeta.modules.utils import get_subdict  # type: ignore
import numpy as np


class PreActBlock(MetaModule):
  '''Pre-activation version of the BasicBlock.'''
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(PreActBlock, self).__init__()
    self.blockConv1 = nn.Sequential(
        MetaConv2d(in_planes,
                   planes,
                   kernel_size=3,
                   stride=stride,
                   padding=1,
                   bias=False), MetaBatchNorm2d(planes), nn.ReLU(inplace=True))
    self.blockConv2 = nn.Sequential(
        MetaConv2d(planes,
                   planes,
                   kernel_size=3,
                   stride=1,
                   padding=1,
                   bias=False), MetaBatchNorm2d(planes))

    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          MetaConv2d(in_planes,
                     planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False))

  def forward(self, x, params=None):
    identity = x
    x = self.blockConv1(x)
    x = self.blockConv2(x)
    x += self.shortcut(identity) if hasattr(self, 'shortcut') else x
    x = nn.ReLU(inplace=True)(x)
    return x


class PreActBottleneck(MetaModule):
  '''Pre-activation version of the BasicBlock.'''
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super(PreActBlock, self).__init__()
    self.bn1 = MetaBatchNorm2d(in_planes)
    self.blockConv1 = nn.Sequential(
        MetaConv2d(in_planes, planes, kernel_size=1, bias=False),
        MetaBatchNorm2d(planes), nn.ReLU(inplace=True))
    self.blockConv2 = nn.Sequential(
        MetaConv2d(planes,
                   planes,
                   kernel_size=3,
                   stride=stride,
                   padding=1,
                   bias=False), MetaBatchNorm2d(planes))
    self.blockConv3 = MetaConv2d(planes,
                                 self.expansion * planes,
                                 kernel_size=1,
                                 stride=stride,
                                 bias=False)

    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          MetaConv2d(in_planes,
                     self.expansion * planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False))

  def forward(self, x, params=None):
    identity = x
    x = self.blockConv1(x)
    x = self.blockConv2(x)
    x = self.blockConv3
    x += self.shortcut(identity) if hasattr(self, 'shortcut') else x
    x = nn.ReLU(inplace=True)(x)
    return x


class PreActResNet(MetaModule):
  def __init__(self, block, num_blocks, num_classes=10):
    super(PreActResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = MetaConv2d(3,
                            64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=False)
    self.bn1 = MetaBatchNorm2d(64)
    self.layer1 = nn.Sequential(PreActBlock(self.in_planes, 64),
                                PreActBlock(self.in_planes, 64))
    self.layer2 = nn.Sequential(PreActBlock(64, 128, 2), PreActBlock(128, 128))
    self.layer3 = nn.Sequential(PreActBlock(128, 256, 2),
                                PreActBlock(256, 256))
    self.layer4 = nn.Sequential(PreActBlock(256, 512, 2),
                                PreActBlock(512, 512))
    self.linear = MetaLinear(512 * block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride, params):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x, params=None, vis=False, vul=False):
    out_1 = F.relu(self.bn1(self.conv1(x)))
    out_2 = self.layer1(out_1)
    out_3 = self.layer2(out_2)
    out_4 = self.layer3(out_3)
    out_5 = self.layer4(out_4)
    out_6 = F.avg_pool2d(out_5, 4)
    out_6 = out_6.view(out_6.size(0), -1)
    out_6 = self.linear(out_6)
    return out_6


def resnet50(n_classes=200):
  return PreActResNet(PreActBottleneck, [3, 4, 6, 3], num_classes=n_classes)


class NoiseResNet3x3Conv(nn.Module):
  def __init__(self, channels=3, custom_init=True):
    super().__init__()
    self.conv_2d_1 = nn.Conv2d(in_channels=channels,
                               out_channels=20,
                               kernel_size=1,
                               stride=1,
                               padding=0)
    self.conv_2d_2 = nn.Conv2d(in_channels=20,
                               out_channels=20,
                               kernel_size=3,
                               stride=1,
                               padding=0)
    self.conv_2d_3 = nn.Conv2d(in_channels=20,
                               out_channels=20,
                               kernel_size=1,
                               stride=1,
                               padding=0)
    self.conv_2d_4 = nn.Conv2d(in_channels=20,
                               out_channels=channels,
                               kernel_size=1,
                               stride=1,
                               padding=0)

  def forward(self, x):
    bs, ch, nx, ny = x.shape
    x = torch.empty((bs, ch, nx + 2, ny + 2), device=x.device).normal_()
    residual = x[:, :, 1:-1, 1:-1]
    x = F.leaky_relu(self.conv_2d_1(x))
    x = F.leaky_relu(self.conv_2d_2(x))
    x = F.leaky_relu(self.conv_2d_3(x))
    x = self.conv_2d_4(x) + residual
    return x


def test():
  net = ResNet50()
  y = net((torch.randn(1, 3, 64, 64)))
  print(y.size())
