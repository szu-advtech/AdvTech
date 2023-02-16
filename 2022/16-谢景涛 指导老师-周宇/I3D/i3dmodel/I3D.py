import torch
import numpy as np
from math import ceil
import torch.nn as nn
#https://stackoverflow.com/questions/37674306/what-is-the-difference-between-same-and-valid-padding-in-tf-nn-max-pool-of-t/49842071#49842071
def get_padding_shape(in_shape, strides,filter_shape):
    in_time=in_shape[0]
    in_height=in_shape[1]
    in_width=in_shape[2]
    out_time=ceil(float(in_time) / float(strides[0]))
    out_height = ceil(float(in_height) / float(strides[1]))
    out_width = ceil(float(in_width) / float(strides[2]))
    pad_along_time = max((out_time - 1) * strides[0] + filter_shape[0] - in_time, 0)
    pad_along_height = max((out_height - 1) * strides[1] +filter_shape[1] - in_height, 0)
    pad_along_width = max((out_width - 1) * strides[2] +filter_shape[2] - in_width, 0)
    pad_front=pad_along_time//2
    pad_back=pad_along_time-pad_front
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left
    return (pad_top,pad_bottom,pad_left,pad_right,pad_front,pad_back)

class Inc(nn.Module):
    def __init__(self,in_channels,out_channels,i,j,k):
        super().__init__()
        self.conv1=nn.Conv3d(in_channels,out_channels[0],kernel_size=1)
        self.bn1 = nn.BatchNorm3d(out_channels[0], eps=0.001, momentum=0.01)

        self.conv2_1=nn.Conv3d(in_channels,out_channels[1],kernel_size=1)
        self.bn2_1 = nn.BatchNorm3d(out_channels[1], eps=0.001, momentum=0.01)

        self.conv2_2=nn.Conv3d(out_channels[1],out_channels[2],kernel_size=3)
        self.bn2_2 = nn.BatchNorm3d(out_channels[2], eps=0.001, momentum=0.01)

        self.conv3_1=nn.Conv3d(in_channels,out_channels[3],kernel_size=1)
        self.bn3_1 = nn.BatchNorm3d(out_channels[3], eps=0.001, momentum=0.01)

        self.conv3_2=nn.Conv3d(out_channels[3],out_channels[4],kernel_size=3)
        self.bn3_2 = nn.BatchNorm3d(out_channels[4], eps=0.001, momentum=0.01)

        self.pool=nn.MaxPool3d(kernel_size=3,stride=1)

        self.conv4=nn.Conv3d(in_channels,out_channels[5],kernel_size=1)
        self.bn4 = nn.BatchNorm3d(out_channels[5], eps=0.001, momentum=0.01)

        self.mo = nn.ConstantPad3d(get_padding_shape((i,j,k), (1, 1, 1), (3, 3, 3)), 0)
    def forward(self,x):
        x1=self.bn1(self.conv1(x))

        x2=self.bn2_2(self.conv2_2(self.mo(self.bn2_1(self.conv2_1(x)))))

        x3=self.bn3_2(self.conv3_2(self.mo(self.bn3_1(self.conv3_1(x)))))

        x4 = self.bn4(self.conv4(self.pool(self.mo(x))))

        x = torch.cat([x1,x2,x3,x4], dim=1)
        return x

class I3D(nn.Module):
    def __init__(self,in_channel:int,num_classes:int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channel,64,kernel_size=(7,7,7),stride=(2,2,2))
        self.BN1 = nn.BatchNorm3d(64,eps=0.001, momentum=0.01)
        self.Mpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 64, kernel_size=1)
        self.BN2 = nn.BatchNorm3d(64,eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv3d(64, 192, kernel_size=3)
        self.BN3 = nn.BatchNorm3d(192,eps=0.001, momentum=0.01)

        self.Inc1=Inc(192,[64,96,128,16,32,32],32,28,28)
        self.Inc2 = Inc(256,[128,128,192,32,96,64],32,28,28)
        self.Inc3 = Inc(128+192+96+64,[192,96,208,16,48,64],16,14,14)
        self.Inc4 = Inc(192+208+48+64,[160,112,224,24,64,64],16,14,14)
        self.Inc5 = Inc(160+224+64+64,[128,128,256,24,64,64],16,14,14)
        self.Inc6 = Inc(128+256+64+64,[112,144,288,32,64,64],16,14,14)
        self.Inc7 = Inc(112+288+64+64,[256,160,320,32,128,128],16,14,14)
        self.Inc8 = Inc(256+320+128+128,[256,160,320,32,128,128],8,7,7)
        self.Inc9 = Inc(256+320+128+128,[384,192,384,48,128,128],8,7,7)

        self.Mpool2 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.Mpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Apool = nn.AvgPool3d(kernel_size=(2,7,7),stride=(1, 1, 1))
        self.conv4 = nn.Conv3d(384+384+128+128, num_classes, kernel_size=1)
        #self.BN4=nn.BatchNorm3d(num_classes)
        self.relu = nn.ReLU()
        self.sofm = nn.Softmax(-1)
        self.mo1=nn.ConstantPad3d(get_padding_shape((64,224,224),(2,2,2),(7,7,7)), 0)
        self.mo2=nn.ConstantPad3d(get_padding_shape((32,112,112),(1, 2, 2),(1, 3, 3)), 0)
        self.mo3 = nn.ConstantPad3d(get_padding_shape((32,56,56), (1, 1, 1), (3, 3, 3)), 0)
        self.mo4 = nn.ConstantPad3d(get_padding_shape((32,28,28), (2, 2, 2), (3, 3, 3)), 0)
        self.mo5 = nn.ConstantPad3d(get_padding_shape((16,14,14), (2, 2, 2), (2, 2, 2)), 0)
    def forward(self,x):
        bn,c,t,w,h=x.shape
        x=self.mo1(x)
        x=self.conv1(x)
        x=self.BN1(x)
        x=self.relu(x)
        x=self.mo2(x)
        x=self.Mpool1(x)
        x=self.conv2(x)
        x = self.BN2(x)
        x = self.relu(x)


        x=self.mo3(x)
        x=self.conv3(x)
        x = self.BN3(x)
        x = self.relu(x)
        x=self.mo2(x)
        x=self.Mpool1(x)
        x=self.Inc1(x)
        x=self.Inc2(x)


        x=self.mo4(x)
        x=self.Mpool2(x)
        x=self.Inc3(x)
        x = self.Inc4(x)
        x = self.Inc5(x)
        x = self.Inc6(x)
        x = self.Inc7(x)

        x=self.mo5(x)
        x=self.Mpool3(x)
        x=self.Inc8(x)
        x = self.Inc9(x)
        x=self.Apool(x)

        x=self.conv4(x)
        #x = self.BN4(x)
        x = self.relu(x)
        out=x
        out = out.squeeze(3)
        out = out.squeeze(3)

        out = out.mean(2)

        out_logits = out

        out_logits=self.sofm(out_logits)

        return out_logits