import torch
import torch.nn as nn
from FBP import voxel_backprojection,siddon_ray_projection
import numpy as np
class RSEB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RSEB, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, padding='same')
        self.PReLU = nn.PReLU()  # 默认值0.25
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding='same')

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.conv3 = nn.Conv2d(out_channel, out_channel, 1, padding='same')
        self.conv4 = nn.Conv2d(out_channel, out_channel, 1, padding='same')
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.PReLU(x)
        x = self.conv2(x)

        x1 = self.gap(x)
        x1 = self.conv3(x1)
        x1 = self.PReLU(x1)
        x1 = self.conv4(x1)
        x1 = self.sigmoid(x1)

        x = x * x1

        return x + inputs


class DownConv(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(DownConv, self).__init__()
        self.down_conv = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(input_channel, out_channel, 3, padding='same'),
            nn.BatchNorm2d(out_channel)
        )

    def forward(self, inputs):
        return self.down_conv(inputs)


class UpConv(nn.Module):
    def __init__(self, input_channel, out_channel,factor=2):
        super(UpConv, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=factor, mode='bilinear'),
            nn.Conv2d(input_channel, out_channel, 3, padding='same'),
            nn.BatchNorm2d(out_channel)

        )

    def forward(self, inputs):
        return self.up_conv(inputs)


class RSEB_Block(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(RSEB_Block, self).__init__()
        self.rseb_block = nn.Sequential(
            RSEB(out_channel, out_channel),
            RSEB(out_channel, out_channel)
        )

    def forward(self, inputs):
        return self.rseb_block(inputs)


class FRB(nn.Module):
    def __init__(self, input_channel, out_channel, n=6):
        super(FRB, self).__init__()
        layers = []
        for i in range(n):
            layers.append(RSEB(input_channel, out_channel))
        layers.append(nn.Conv2d(out_channel, out_channel, 3, padding='same'))
        layers.append(nn.BatchNorm2d(out_channel))
        self.model = nn.Sequential(*layers)

    def forward(self, inputs):
        return self.model(inputs)


class UFNet(nn.Module):
    def __init__(self, input_channel, out_channel):
        super(UFNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, out_channel, 3, padding='same')#
        self.rseb1 = RSEB(out_channel, out_channel)

        self.rseb_block1 = RSEB_Block(out_channel, out_channel)

        self.down_conv1 = DownConv(out_channel, 2*out_channel)

        self.rseb_block2 = RSEB_Block(2*out_channel, 2*out_channel)

        self.down_conv2 = DownConv(2*out_channel, 4 * out_channel)

        self.rseb_block3 = RSEB_Block(4 * out_channel, 4 * out_channel)

        self.rseb_block4 = RSEB_Block(4 * out_channel, 4 * out_channel)

        self.up_conv1 = UpConv(4*out_channel, 2*out_channel)
        self.rseb_block5 = RSEB_Block(2*out_channel, 2*out_channel)
        self.up_conv2 = UpConv(2*out_channel, out_channel)
        self.rseb_block6 = RSEB_Block(out_channel, out_channel)

        self.rseb_connect1 = RSEB(2*out_channel, 2*out_channel)
        self.rseb_connect2 = RSEB(out_channel, out_channel)

        self.p1 = nn.Conv2d(out_channel, out_channel, 3, padding='same')
        self.p2 = nn.Conv2d(1, 1, 3, padding='same')
        self.p_fu = nn.Conv2d(out_channel, 1, 3, padding='same')
        self.sigmoid = nn.Sigmoid()

        # Stage-2
        self.conv2 = nn.Conv2d(input_channel, out_channel, 3, padding='same')#
        self.rseb2 = RSEB(out_channel, out_channel)
        self.po1 = FRB(2*out_channel, 2*out_channel)
        self.pe3 = UpConv(4*out_channel, 2*out_channel,4)
        self.pd3 = UpConv(4*out_channel, 2*out_channel,4)
        self.po2 = FRB(2*out_channel, 2*out_channel)
        self.pe2 = UpConv(2*out_channel, 2*out_channel)
        self.pd2 = UpConv(2*out_channel, 2*out_channel)
        self.po3 = FRB(2*out_channel, 2*out_channel)
        self.pe1 = nn.Conv2d(out_channel, 2*out_channel, 3, padding='same')
        self.pd1 = nn.Conv2d(out_channel, 2*out_channel, 3, padding='same')
        self.p_fo = nn.Conv2d(2*out_channel, 1, 3, padding='same')

    def forward(self, inputs):
        x = self.conv1(inputs)
        x_in1 = inputs[:, 0:1, :, :]
        x = self.rseb1(x)
        f_e1 = self.rseb_block1(x)

        x = self.down_conv1(f_e1)

        f_e2 = self.rseb_block2(x)

        x = self.down_conv2(f_e2)
        f_e3 = self.rseb_block3(x)

        f_d3 = self.rseb_block4(f_e3)
        x = self.up_conv1(f_d3)

        x = x + self.rseb_connect1(f_e2)

        f_d2 = self.rseb_block5(x)
        x = self.up_conv2(f_d2)

        x = x + self.rseb_connect2(f_e1)

        f_d1 = self.rseb_block6(x)
        x_u = self.p_fu(f_d1) + x_in1

        f_att = f_d1 + self.p1(f_d1) * self.sigmoid(self.p2(x_u))

        # stage-2
        x2 = self.rseb2(self.conv2(inputs))
        f_init = torch.cat((x2, f_att), 1)

        f_o1 = self.po1(f_init) + self.pe3(f_e3) + self.pd3(f_d3)


        f_o2 = self.po2(f_o1) + self.pe2(f_e2) + self.pd2(f_d2)

        f_o3 = self.po3(f_o2) + self.pe1(f_e1) + self.pd1(f_d1)
        x_final = x_in1 + self.p_fo(f_o3)

        return x_u, x_final
def Normalize(data):
    mins, maxs = np.min(data), np.max(data)
    img_nor = (data - mins) / (maxs  - mins )
    return img_nor

class DuDoUFNet(nn.Module):
    def __init__(self,m1,m2,channel):
        super(DuDoUFNet, self).__init__()
        self.model1 = UFNet(m1,channel)
        self.model2 = UFNet(m2,channel * 2)
        geo_mode = 'fanflat'  # or 'parallel'
        angle_range = {'fanflat': 2 * np.pi, 'parallel': np.pi}
        geo_full = {'nVoxelX': 416, 'sVoxelX': 340.0192, 'dVoxelX': 0.6641,
                    'nVoxelY': 416, 'sVoxelY': 340.0192, 'dVoxelY': 0.6641,
                    'nDetecU': 640, 'sDetecU': 504.0128, 'dDetecU': 0.6848,
                    'views': 640, 'slices': 1,
                    'DSD': 600.0, 'DSO': 550.0, 'DOD': 50.0,
                    'start_angle': 0.0, 'end_angle': angle_range[geo_mode],
                    'mode': geo_mode
                    }
        self.fan_bp = voxel_backprojection(geo_full)
        self.fan_fp = siddon_ray_projection(geo_full)

    def forward(self,S_ldma,M_proj,X_ldma,M):
        stage1_inputs = torch.cat((S_ldma, M_proj), 1).cuda()

        S_o, S_u  = self.model1(stage1_inputs)

        X_o = self.fan_bp(S_o)
        stage_2_input = torch.cat([X_ldma, X_o, M], dim=1)
        X_final, X_u = self.model2(stage_2_input)

        return S_o, S_u, X_final, X_u,X_o