# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/8 16:19
 @name: 
 @desc:   Fig. 3 and 4 in citation
"""
import torch.nn as nn
import torch
from einops import rearrange


class CFE(nn.Module):
    def __init__(self, channels, E=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=E//2,
                               kernel_size=(1, 3), stride=(1, 1), padding='same')
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=E//2,
                               kernel_size=(1, 5), stride=(1, 1), padding='same')
        self.bn = nn.BatchNorm2d(num_features=E)
        self.elu = nn.ELU()
        self.conv3 = nn.Conv2d(in_channels=E, out_channels=channels,
                               kernel_size=(1, 1), stride=(1, 1), padding='same')

    def forward(self, x):
        # [b, c, p, t]
        x1 = self.conv1(x)  # [b, E/2, p, t]
        x2 = self.conv2(x)  # [b, E/2, p, t]
        x = torch.cat((x1, x2), dim=1)  # [b, E, p, t]
        x = self.bn(x)
        x = self.elu(x)
        x = self.conv3(x)  # [b, c, p, t]
        return x


class MHA(nn.Module):
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.h = num_heads
        self.d = channels // num_heads
        # scale factor
        self.scale = self.d ** -0.5

        self.conv_qkv = nn.Conv2d(in_channels=channels, out_channels=3*channels, kernel_size=(1, 1), stride=(1, 1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # [b, c, p, t]
        qkv = self.conv_qkv(x)  # [b, c, p, t] -> [b, 3*c, p, t]
        q, k, v = rearrange(qkv, 'b (qkv h d) p t -> qkv b h d p t', qkv=3, h=self.h, d=self.d)
        q = rearrange(q, 'b h d p t -> b h p (d t)')
        k = rearrange(k, 'b h d p t -> b h (d t) p')
        v = rearrange(v, 'b h d p t -> b h p (d t)')

        dots = torch.matmul(q, k) * self.scale  # [b, h, p, p]
        attn = self.softmax(dots)

        out = torch.matmul(attn, v)  # [b, h, p, (dt)]
        out = rearrange(out, 'b h p (d t) -> b (h d) p t', h=self.h, d=self.d)
        return out


class CTBlock(nn.Module):
    def __init__(self, channels, num_heads, E):
        super().__init__()
        self.mha = MHA(channels=channels, num_heads=num_heads)
        self.bn1 = nn.BatchNorm2d(num_features=channels)
        self.cfe = CFE(channels=channels, E=E)
        self.bn2 = nn.BatchNorm2d(num_features=channels)

    def forward(self, x):
        # [b, c, p=m*m, T]
        x = torch.add(self.mha(x), x)
        x = self.bn1(x)
        x = torch.add(self.cfe(x), x)
        x = self.bn2(x)
        return x

