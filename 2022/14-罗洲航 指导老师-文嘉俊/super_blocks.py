import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import OrderedDict
import numpy as np

class Conv3D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm3d(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.op(x)

class Conv2D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(Conv2D, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_out, (kernel_size, kernel_size, 1),
                      stride=stride, padding=(padding, padding, 0), bias=False),
            nn.BatchNorm3d(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.op(x)

class Conv1D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(Conv1D, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_out, (1, 1, kernel_size), stride=stride, padding=(0, 0, padding), bias=False),
            nn.BatchNorm3d(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.op(x)

class ConvP3D(nn.Module):
    def __init__(self, C_in, C_out, kernel_size, stride, padding):
        super(ConvP3D, self).__init__()
        self.op = nn.Sequential(
            nn.Conv3d(C_in, C_out, kernel_size = (kernel_size, kernel_size, 1),
                      stride = stride, padding = (padding, padding, 0), bias = False),
            nn.Conv3d(C_out, C_out, kernel_size = (1, 1, kernel_size),
                      stride = 1, padding = (0, 0, padding), bias = False),
            nn.BatchNorm3d(C_out),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.op(x)

class SuperNet(nn.Module):
    def __init__(self, C_in, kernel_size, stride, padding, cfg=None):
        super(SuperNet, self).__init__()
        self.ops = nn.ModuleList()
        if cfg is None:
            self.cfg = [1, 1, 1]
        else:
            self.cfg = np.where(cfg == np.amax(cfg, axis=0), 1, 0)

        '''if self.cfg[0]:
            self.ops.append(Conv3D(C_in, 1, kernel_size, stride, padding))'''
        if self.cfg[0]:
            self.ops.append(Conv2D(C_in, 1, kernel_size,stride, padding))
        if self.cfg[1]:
            self.ops.append(Conv1D(C_in, 1, kernel_size, stride, padding))
        if self.cfg[2]:
            self.ops.append(ConvP3D(C_in, 1, kernel_size, stride, padding))

    def forward(self, x, search=True):
        if search:
            weight = []
            for m in self.ops.modules():
                if isinstance(m, nn.BatchNorm3d):
                    weight.append(m.weight.data.abs().clone())
            weight = F.softmax(torch.Tensor(weight), dim=-1)
            out = sum(w * op(x) for w, op in zip(weight, self.ops))
        else:
            out = self.ops[0](x)

        return out