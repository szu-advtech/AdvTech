import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from super_blocks import *

class Bottleneck(nn.Module):
    def __init__(self, C_in, C_out, stride, cfg=None):
        super(Bottleneck, self).__init__()
        if cfg is None:
            self.cfg = [None] * C_out
        else:
            self.cfg = cfg

        self.convs = nn.ModuleList()
        for ch in range(C_out):
            self.convs.append(Super0Net(C_in, kernel_size=3, stride=stride, padding=1, cfg=self.cfg[ch]))

    def forward(self, x, search=True):
        xs = []
        for conv in self.convs:
            xs.append(conv(x, search))
        x = torch.cat(xs, dim=1)
        return x

class CostReg(nn.Module):

    def __init__(self, cfg=None):

        super(CostReg, self).__init__()
        if cfg is None:
            self.cfg = [None] * 7
        else:
            self.cfg = cfg

        self.layer1 = Bottleneck(32, 8, stride=1, cfg=self.cfg[0])

        self.layer2 = Bottleneck(8, 16, stride=2, cfg=self.cfg[1])
        self.layer3 = Bottleneck(16, 16, stride=1, cfg=self.cfg[2])

        self.layer4 = Bottleneck(16, 32, stride=2, cfg=self.cfg[3])
        self.layer5 = Bottleneck(32, 32, stride=1, cfg=self.cfg[4])

        self.layer6 = Bottleneck(32, 64, stride=2, cfg=self.cfg[5])
        self.layer7 = Bottleneck(64, 64, stride=1, cfg=self.cfg[6])

        self.upsample1 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, search=True):
        block1 = self.layer1(x, search)

        block2 = self.layer2(block1, search)
        block2 = self.layer3(block2, search)

        block3 = self.layer4(block2, search)
        block3 = self.layer5(block3, search)

        block4 = self.layer6(block3, search)
        block4 = self.layer7(block4, search)

        out = block3 + self.upsample1(block4)
        out = block2 + self.upsample2(out)
        out = block1 + self.upsample3(out)

        out = self.prob(out)

        return out