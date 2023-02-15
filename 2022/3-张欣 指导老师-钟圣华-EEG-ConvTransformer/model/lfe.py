# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/8 15:39
 @name: 
 @desc: Implementation of Fig. 2 of Session 3.2 in citation
"""

import torch.nn as nn
import torch


class LocFeaExtractor(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.c = channels
        self.conv3d_3 = nn.Conv3d(in_channels=1, out_channels=channels//2, bias=True,
                                  kernel_size=(8, 8, 3), stride=(4, 4, 1), padding=(0, 0, 1))
        self.conv3d_5 = nn.Conv3d(in_channels=1, out_channels=channels//2, bias=True,
                                  kernel_size=(8, 8, 5), stride=(4, 4, 1), padding=(0, 0, 2))

        self.bn = nn.BatchNorm3d(num_features=channels)
        self.elu = nn.ELU()

    def forward(self, x):
        [b, _, _, _, t] = x.shape

        # [b, 1,  M, M, T]
        x_3 = self.conv3d_3(x)  # [b, c/2, m, m, T]
        x_5 = self.conv3d_5(x)  # [b, c/2, m, m, T]
        x = torch.cat((x_3, x_5), dim=1)  # [b, c, m, m, T]

        x = self.bn(x)
        x = self.elu(x)
        x = torch.reshape(x, [b, self.c, -1, t])  # [b, c, m*m, T]
        return x




