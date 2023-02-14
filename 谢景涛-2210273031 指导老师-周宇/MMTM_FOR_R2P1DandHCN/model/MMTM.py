import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _triple
class SpatioTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()

        stride = _triple(stride)
        padding = _triple(padding)
        kernel_size = _triple(kernel_size)


        spatial_stride = [1, stride[1], stride[2]]
        spatial_padding = [0, padding[1], padding[2]]
        spatial_kernel_size = [1, kernel_size[1], kernel_size[2]]

        temporal_stride = [stride[0], 1, 1]
        temporal_padding = [padding[0], 0, 0]
        temporal_kernel_size = [kernel_size[0], 1, 1]


        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels) /
                                           (kernel_size[1] * kernel_size[2] * in_channels + kernel_size[0] * out_channels)))


        self.spatial_conv = nn.Conv3d(in_channels, intermed_channels, spatial_kernel_size,
                                      stride=spatial_stride, padding=spatial_padding, bias=bias)
        self.bn = nn.BatchNorm3d(intermed_channels)
        self.relu = nn.ReLU()


        self.temporal_conv = nn.Conv3d(intermed_channels, out_channels, temporal_kernel_size,
                                       stride=temporal_stride, padding=temporal_padding, bias=bias)

    def forward(self, x):
        x = self.relu(self.bn(self.spatial_conv(x)))
        x = self.temporal_conv(x)
        return x



def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3x3 Factored Spatial-Temporal convolution with padding."""
    return SpatioTemporalConv(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)

class BaseBasicBlock(nn.Module):
    expansion = 1
    Conv3d = staticmethod(conv3x3x3)

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BaseBasicBlock, self).__init__()
        self.conv1 = self.Conv3d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = self.Conv3d(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out

class BasicBlock(BaseBasicBlock):
    Conv3d = staticmethod(conv3x3x3)
def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1),
        out.size(2), out.size(3), out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class hcn_point(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.conv1=nn.Conv2d(in_channel,64,1)
        self.conv2=nn.Conv2d(64,32,kernel_size=(3,1),padding=(1,0))
        self.relu = nn.ReLU()
    def forward(self,x):
        x=self.conv1(x)
        x=self.relu(x)
        x=self.conv2(x)
        return x

class hcn_cooccurrence(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channel, 32, 3,padding=1)
        self.conv4 = nn.Conv2d(32, 64, 3,padding=1)
        self.layer = nn.MaxPool2d(2, stride=2)
        self.drop4 = nn.Dropout(p=0.5)
    def forward(self,x):
        x=self.conv3(x)
        x=self.layer(x)
        x=self.conv4(x)
        x=self.layer(x)
        x=self.drop4(x)
        return x
def init_weights(m):
 
 if type(m) == nn.Linear:
   #print(m.weight)
   pass
 else:
   #print('error')
   pass

class flusion(nn.Module):
    def __init__(self,A_channel,B_channel):
        super().__init__()
        self.linear=nn.Linear(A_channel+B_channel,(A_channel+B_channel)//4)
        self.linearA=nn.Linear((A_channel+B_channel)//4,A_channel)
        self.linearB=nn.Linear((A_channel+B_channel)//4,B_channel)
        self.softmax=nn.Softmax(dim=1)

    def forward(self,x):
        [A,B]=x
        self.AVGpool = nn.AvgPool3d(kernel_size=(A.shape[-3],A.shape[-2],A.shape[-1]))
        self.AVGpool2=nn.AvgPool2d(kernel_size=(B.shape[-2],B.shape[-1]))
        SA=self.AVGpool(A)
        SB=self.AVGpool2(B)
        SA=SA.squeeze(-1).squeeze(-1).squeeze(-1)
        SB=SB.squeeze(-1).squeeze(-1)
        if A.shape[0]*2==B.shape[0]:
            SA=SA.repeat(2,1)
        C=torch.cat((SA,SB),dim=1)
        Z=self.linear(C)
        EA=self.linearA(Z)
        EB=self.linearB(Z)
        if A.shape[0]*2==B.shape[0]:
            EA=EA[0:A.shape[0]]
        temp=2*self.softmax(EA)
        temp=temp.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        A=(temp*A)
        temp = 2 * self.softmax(EB)
        temp = temp.unsqueeze(2).unsqueeze(3)
        B=(temp*B)
        return A,B

class MMTM(nn.Module):

    Conv3d = SpatioTemporalConv

    def __init__(self, block, layers,inpointchannel,injointchanel, shortcut_type='B', num_classes=339):
        self.inplanes = 64
        super(MMTM, self).__init__()
        self.conv1 = self.Conv3d(1, 64, kernel_size=(3,7,7), stride=(1, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.hcn_point_S = hcn_point(inpointchannel)
        self.hcn_point_M = hcn_point(inpointchannel)
        self.hcn_cooccurrence_S = hcn_cooccurrence(injointchanel)
        self.hcn_cooccurrence_M = hcn_cooccurrence(injointchanel)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)
        self.layer = nn.MaxPool2d(2, stride=2)
        self.relu_s = nn.ReLU()
        self.fc7 = nn.Linear(1024, 256)
        self.fc8 = nn.Linear(256, num_classes)
        self.sofm=nn.Softmax(-1)

        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.5)
        self.drop7 = nn.Dropout(p=0.5)
        self.flu1 = flusion(256, 1024)
        self.flu2 = flusion(512, 256)


    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    self.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def forward(self, x):
        [skeleten, depth] = x
        # ____________________________________________________
        depth = self.conv1(depth)
        depth = self.bn1(depth)
        depth = self.relu(depth)
        depth = self.maxpool(depth)
        depth = self.layer1(depth)
        depth = self.layer2(depth)
        depth = self.layer3(depth)
        # ____________________________________________________
        batch, pople, frame, joint, channel = skeleten.shape
        skeleten = skeleten.view(-1, frame, joint, channel)
        skeleten = skeleten.permute(0, 3, 1, 2).contiguous()
        vel1 = skeleten[:, :, :1] * 0
        vel2 = skeleten[:, :, 1:] - skeleten[:, :, :-1]
        m = torch.cat((vel1, vel2), dim=2)
        skeleten = self.hcn_point_S(skeleten)
        skeleten = skeleten.permute(0, 3, 2, 1)
        skeleten = self.hcn_cooccurrence_S(skeleten)
        m = self.hcn_point_M(m)
        m = m.permute(0, 3, 2, 1)
        m = self.hcn_cooccurrence_M(m)
        c = torch.cat((skeleten, m), dim=1)
        c = self.conv5(c)
        c = self.layer(c)
        c = self.relu(c)
        c = self.drop5(c)
        c = self.conv6(c)
        c = self.layer(c)
        c = self.relu(c)
        c = c.view(batch, pople, -1)  # N M D
        c = c.max(dim=1)[0]  # N D
        c = self.drop6(c)
        c = c.unsqueeze(2).unsqueeze(3)
        # ____________________________________________________
        # ____________________________________________________
        depth, c = self.flu1(depth, c)
        # ____________________________________________________
        # ____________________________________________________
        depth = self.layer4(depth)
        # ____________________________________________________
        c = c.squeeze(-1).squeeze(-1)
        c = self.fc7(c)
        c = self.relu(c)
        c = self.drop7(c)
        c = c.unsqueeze(2).unsqueeze(3)
        # ____________________________________________________
        # ____________________________________________________
        depth, c = self.flu2(depth, c)
        # ____________________________________________________
        # ____________________________________________________
        c = c.squeeze(-1).squeeze(-1)
        c = self.fc8(c)


        depth = self.avgpool(depth)

        depth = depth.view(depth.size(0), -1)
        depth = self.fc(depth)
        c = self.sofm(c)
        depth=self.sofm(depth)
        out_logits = self.sofm(depth + c)

        return out_logits



def mmtm(inpointchannel=3,injointchanel=25,numclasses=60):
    model = MMTM(BasicBlock, [2, 2, 2, 2],inpointchannel,injointchanel,num_classes=numclasses)
    return model











