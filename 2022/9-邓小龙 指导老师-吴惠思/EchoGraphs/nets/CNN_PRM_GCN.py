import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from nets.encoders import image_encoder
from typing import List, Tuple

"""
Code for spiral convolutions adapted from 
https://github.com/gbouritsas/Neural3DMM
Bouritsas, G. et al.: Neural 3D Morphable Models: Spiral Convolutional Networks for 3D Shape Representation Learning and Generation, ICCV, 2019
"""
class SpiralConv(nn.Module):
    def __init__(self, in_channels, out_channels, indices, dim=1, tsteps=1):
        super(SpiralConv, self).__init__()
        self.dim = dim
        self.indices = indices
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.seq_length = indices.size(1)
        if tsteps==2:
            self.cycle = 1
        elif tsteps==1:
            self.cycle = 0
        else:
            self.cycle = 2

        self.layer = nn.Linear(in_channels * (self.seq_length+self.cycle), out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.layer.weight)
        torch.nn.init.constant_(self.layer.bias, 0)

    def forward(self, x):
        n_nodes, _ = self.indices.size()
        if x.dim() == 2:
            x = torch.index_select(x, 0, self.indices.view(-1))
            x = x.view(n_nodes, -1)
            x = self.layer(x)
        elif x.dim() == 3:
            bs = x.size(0)
            x = torch.index_select(x, self.dim, self.indices.view(-1))
            x = x.view(bs, n_nodes, -1)
            x = self.layer(x)
        elif x.dim() == 4:
            bs = x.size(0)
            frames = x.size(3)
            pre_idx = torch.roll(torch.arange(0, frames), 1)
            post_idx = torch.roll(torch.arange(0, frames), 1)
            for idx in range(0,frames):

                x1 = torch.index_select(x[:,:,:,idx], self.dim, self.indices.view(-1))
                x1 = torch.cat((x1,torch.index_select(x[:,:,:,pre_idx[idx]], self.dim, self.indices[0].view(-1))),dim=1)
                if(frames>3):
                    x1 = torch.cat((x1,torch.index_select(x[:,:,:,post_idx[idx]], self.dim, self.indices[0].view(-1))), dim=1)
                x1 = x1.view(bs, n_nodes, -1)

                x1 = self.layer(x1)
                if idx==0:
                    x_out= x1.unsqueeze(3)
                else:
                    x_out = torch.cat((x_out, x1.unsqueeze(3)), dim=3)
            x= x_out
        else:
            raise RuntimeError(
                'x.dim() is expected to be 2, 3 and 4, but received {}'.format(
                    x.dim()))
        return x

    def __repr__(self):
        return '{}({}, {}, seq_length={})'.format(self.__class__.__name__,
                                                  self.in_channels,
                                                  self.out_channels,
                                                  self.seq_length)


class kpts_decoder(nn.Module):

    def __init__(self, features_size, kpt_channels, gcn_channels, num_kpts=8, tsteps=1, is_gpu=True):

        super(kpts_decoder, self).__init__()

        self.kpt_channels = kpt_channels
        self.gcn_channels = gcn_channels

        # construct nodes for graph CNN decoder:
        self.num_nodes = num_kpts

        # construct edges for graph CNN decoder:
        adjacency = self.create_graph(self.num_nodes)

        # init GCN:
        self.init_gcn(adjacency, features_size, tsteps, is_gpu)

    def create_graph(self, num_nodes: int) -> np.ndarray:
        adjacency = []
        for ii in range(num_nodes):
            x = list(range(num_nodes))
            x.insert(0, x.pop(ii))
            adjacency.append(x)
        adjacency = np.array(adjacency)

        return adjacency

    def init_gcn(self, adjacency: np.ndarray, features_size: int, tsteps: int, is_gpu: bool):

        self.spiral_indices = torch.from_numpy(adjacency)
        if not is_gpu:
            self.spiral_indices = self.spiral_indices.cpu()
        else:
            self.spiral_indices = self.spiral_indices.cuda()

        #self.regression_layer = nn.Sequential(nn.Linear(self.gcn_channels[0]*num_kpts,1), nn.Sigmoid())
        # construct graph CNN layers:
        self.decoder_layers = nn.ModuleList()
        self.decoder_layers.append(nn.Linear(features_size, self.num_nodes * self.gcn_channels[-1]))
        for idx in range(len(self.gcn_channels)):
            if idx == 0:
                self.decoder_layers.append(
                    SpiralConv(self.gcn_channels[-idx - 1],
                               self.gcn_channels[-idx - 1],
                               self.spiral_indices, tsteps=tsteps))
            else:
                self.decoder_layers.append(
                    SpiralConv(self.gcn_channels[-idx], self.gcn_channels[-idx - 1],
                               self.spiral_indices, tsteps=tsteps))
        self.decoder_layers.append(
            SpiralConv(self.gcn_channels[0], self.kpt_channels, self.spiral_indices, tsteps=tsteps))

    def forward(self, x):
        num_layers = len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            if i == 0: # 第一层
                x = layer(x)
                x = x.view(-1, self.num_nodes, self.gcn_channels[-1])
            elif i != num_layers - 1: # 不是最后一层
                x = F.elu(layer(x))
            else:
                x = layer(x)
        return x

class CNNPRMGCN(nn.Module):
    def __init__(self, kpt_channels, gcn_channels, backbone=18, num_kpts=8, is_gpu=True):

        super(CNNPRMGCN, self).__init__()

        self.image_encoder = image_encoder(backbone=backbone)

        self.prm = PRM(output_chl_num=2048)

        self.kpts_decoder = kpts_decoder(features_size=self.image_encoder.img_feature_size,
                                         kpt_channels=kpt_channels,
                                         gcn_channels=gcn_channels,
                                         num_kpts=num_kpts,
                                         is_gpu=is_gpu)

    def forward(self, x):
        features = self.image_encoder(x)
        features = self.prm(features)
        kpts = self.kpts_decoder(features)
        return kpts

"""
code for PRM adapted from 
https://github.com/caiyuanhao1998/RSN
Learning Delicate Local Representations for Multi-Person Pose Estimation (ECCV 2020 Spotlight)
"""

class conv_bn_relu(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, 
            has_bn=True, has_relu=True, efficient=False,groups=1):
        super(conv_bn_relu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size,
                stride=stride, padding=padding,groups=groups)
        self.has_bn = has_bn
        self.has_relu = has_relu
        self.efficient = efficient
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        def _func_factory(conv, bn, relu, has_bn, has_relu):
            def func(x):
                x = conv(x)
                if has_bn:
                    x = bn(x)
                if has_relu:
                    x = relu(x)
                return x
            return func 

        func = _func_factory(
                self.conv, self.bn, self.relu, self.has_bn, self.has_relu)

        # if self.efficient:
        #     x = checkpoint(func, x)
        # else:
        #     x = func(x)

        x = func(x)
        return x

class PRM(nn.Module):

    def __init__(self, output_chl_num, efficient=False):
        super(PRM, self).__init__()
        self.output_chl_num = output_chl_num
        self.conv_bn_relu_prm_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=3,
                stride=1, padding=1, has_bn=True, has_relu=True,
                efficient=efficient) 
        self.conv_bn_relu_prm_2_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.conv_bn_relu_prm_2_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.sigmoid2 = nn.Sigmoid()
        self.conv_bn_relu_prm_3_1 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=1,
                stride=1, padding=0, has_bn=True, has_relu=True,
                efficient=efficient)
        self.conv_bn_relu_prm_3_2 = conv_bn_relu(self.output_chl_num, self.output_chl_num, kernel_size=9,
                stride=1, padding=4, has_bn=True, has_relu=True,
                efficient=efficient,groups=self.output_chl_num)
        self.sigmoid3 = nn.Sigmoid()
        self.pool2d = nn.AdaptiveAvgPool2d(output_size=(1,1))
        self.flatten = nn.Flatten(start_dim=1)

    def forward(self, x):
        out = self.conv_bn_relu_prm_1(x)
        out_1 = out
        out_2 = torch.nn.functional.adaptive_avg_pool2d(out_1, (1,1))
        out_2 = self.conv_bn_relu_prm_2_1(out_2)
        out_2 = self.conv_bn_relu_prm_2_2(out_2)
        out_2 = self.sigmoid2(out_2)
        out_3 = self.conv_bn_relu_prm_3_1(out_1)
        out_3 = self.conv_bn_relu_prm_3_2(out_3)
        out_3 = self.sigmoid3(out_3)
        out = out_1.mul(1 + out_2.mul(out_3))
        out = self.pool2d(out)
        out = self.flatten(out)
        return out


if __name__ == '__main__':
    num_batches = 64
    num_kpts = 40
    kpt_channels = 2 # kpts dim
    img = torch.rand(num_batches, 3, 112, 112).cpu()
    #m = CNN_GCN(kpt_channels=kpt_channels, gcn_channels=[16, 32, 32, 48], backbone='mobilenet2_quantize', num_kpts=num_kpts)

    m = CNNPRMGCN(kpt_channels=2, gcn_channels=[4, 8, 8, 16, 16, 32, 32, 48], backbone='resnet50', num_kpts=num_kpts, is_gpu=False)

    m = m.cpu()


    o = m(img).cuda()
    kpts = torch.rand(num_batches, num_kpts, kpt_channels).cuda()
    print(o.size())
    print(kpts.size())
    l2loss = nn.L1Loss()
    print(l2loss(o, kpts))
    # optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    # optimizer.step()
    pass
