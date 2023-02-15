import torch
import numpy as np
import math
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



def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def normal_init(module: nn.Module,
                mean: float = 0,
                std: float = 1,
                bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class hcn_point(nn.Module):
    def __init__(self,in_channel):
        super().__init__()
        self.conv1=nn.Conv2d(in_channel,64,1)
        self.conv2=nn.Conv2d(64,32,kernel_size=(3,1),padding=(1,0))
        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
    def forward(self,x):
        x=self.conv3(x)
        x=self.layer(x)
        x=self.conv4(x)
        x=self.layer(x)
        x=self.drop4(x)
        return x

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

        

class mmtm(nn.Module):
    def __init__(self, inpointchannel, injointchanel,invideochanle, num_classes):
        super().__init__()
        self.conv1 = nn.Conv3d(invideochanle, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2))
        self.BN1 = nn.BatchNorm3d(64,eps=0.001, momentum=0.01)
        self.Mpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 64, kernel_size=1)
        self.BN2 = nn.BatchNorm3d(64,eps=0.001, momentum=0.01)
        self.conv3 = nn.Conv3d(64, 192, kernel_size=3)
        self.BN3 = nn.BatchNorm3d(192,eps=0.001, momentum=0.01)
        self.Inc1 = Inc(192, [64, 96, 128, 16, 32, 32], 32, 28, 28)
        self.Inc2 = Inc(256, [128, 128, 192, 32, 96, 64], 32, 28, 28)
        self.Inc3 = Inc(128 + 192 + 96 + 64, [192, 96, 208, 16, 48, 64], 16, 14, 14)
        self.Inc4 = Inc(192 + 208 + 48 + 64, [160, 112, 224, 24, 64, 64], 16, 14, 14)
        self.Inc5 = Inc(160 + 224 + 64 + 64, [128, 128, 256, 24, 64, 64], 16, 14, 14)
        self.Inc6 = Inc(128 + 256 + 64 + 64, [112, 144, 288, 32, 64, 64], 16, 14, 14)
        self.Inc7 = Inc(112 + 288 + 64 + 64, [256, 160, 320, 32, 128, 128], 16, 14, 14)#in
        self.Inc8 = Inc(256 + 320 + 128 + 128, [256, 160, 320, 32, 128, 128], 8, 7, 7)#in
        self.Inc9 = Inc(256 + 320 + 128 + 128, [384, 192, 384, 48, 128, 128], 8, 7, 7)#in
        self.Mpool2 = nn.MaxPool3d(kernel_size=3, stride=2)
        self.Mpool3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.Apool = nn.AvgPool3d(kernel_size=(2, 7, 7), stride=(1, 1, 1))
        self.conv4 = nn.Conv3d(384 + 384 + 128 + 128, num_classes, kernel_size=1)
        #self.BN4 = nn.BatchNorm3d(num_classes)
        self.relu = nn.ReLU()
        self.sofm = nn.Softmax(-1)
        self.mo1 = nn.ConstantPad3d(get_padding_shape((64, 224, 224), (2, 2, 2), (7, 7, 7)), 0)
        self.mo2 = nn.ConstantPad3d(get_padding_shape((32, 112, 112), (1, 2, 2), (1, 3, 3)), 0)
        self.mo3 = nn.ConstantPad3d(get_padding_shape((32, 56, 56), (1, 1, 1), (3, 3, 3)), 0)
        self.mo4 = nn.ConstantPad3d(get_padding_shape((32, 28, 28), (2, 2, 2), (3, 3, 3)), 0)
        self.mo5 = nn.ConstantPad3d(get_padding_shape((16, 14, 14), (2, 2, 2), (2, 2, 2)), 0)

        self.hcn_point_S = hcn_point(inpointchannel)
        self.hcn_point_M = hcn_point(inpointchannel)
        self.hcn_cooccurrence_S = hcn_cooccurrence(injointchanel)
        self.hcn_cooccurrence_M = hcn_cooccurrence(injointchanel)
        self.conv5 = nn.Conv2d(128, 128, 3, padding=1)#in
        self.conv6 = nn.Conv2d(128, 256, 3, padding=1)#in
        self.layer = nn.MaxPool2d(2, stride=2)
        self.relu = nn.ReLU()
        self.fc7 = nn.Linear(1024, 256)#in
        self.fc8 = nn.Linear(256, num_classes)
        # self.sofm=nn.Softmax()

        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.5)
        self.drop7 = nn.Dropout(p=0.5)
       # self.flu1=flusion((16,14,14),832,(4,4),128)
       # self.flu2=flusion((8,7,7),832,1,1024)
       # self.flu3=flusion((8,7,7),1024,1,256)
        self.flu1=flusion(832,128)
        self.flu2=flusion(832,1024)
        self.flu3=flusion(1024,256)
        self.linearflu=nn.Linear(2*num_classes,num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

    def forward(self,x):
        [skeleten,video]=x
        bn, c, t, w, h = video.shape
        video = self.mo1(video)
        video = self.conv1(video)
        video = self.BN1(video)
        video = self.relu(video)
        video = self.mo2(video)
        video = self.Mpool1(video)
        video = self.conv2(video)
        video = self.BN2(video)
        video = self.relu(video)
        video = self.mo3(video)
        video = self.conv3(video)
        video = self.BN3(video)
        video = self.relu(video)
        video = self.mo2(video)
        video = self.Mpool1(video)
        video = self.Inc1(video)
        video = self.Inc2(video)
        video = self.mo4(video)
        video = self.Mpool2(video)
        video = self.Inc3(video)
        video = self.Inc4(video)
        video = self.Inc5(video)
        video = self.Inc6(video)
#____________________________________________________
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
#_______________________________________________________________
        c = self.conv5(c)
        c = self.layer(c)
        c = self.relu(c)
        c = self.drop5(c)
        video = self.Inc7(video)

#do something
        video,c=self.flu1([video,c])

        c = self.conv6(c)
        c = self.layer(c)
        c = self.relu(c)
        c = c.view(batch, pople, -1)  # N M D
        c = c.max(dim=1)[0]  # N D
        c = self.drop6(c)
        c = c.unsqueeze(2).unsqueeze(3)
        video = self.mo5(video)
        video = self.Mpool3(video)
        video = self.Inc8(video)

#do something
        video, c = self.flu2([video, c])

        c = c.squeeze(-1).squeeze(-1)
        c = self.fc7(c)
        c = self.relu(c)
        c = self.drop7(c)
        c = c.unsqueeze(2).unsqueeze(3)
        video = self.Inc9(video)

# do something
        video, c = self.flu3([video, c])
        c = c.squeeze(-1).squeeze(-1)
#______________________________________________________

        c = self.fc8(c)
        video = self.Apool(video)

        video = self.conv4(video)
        #video = self.BN4(video)
        video = self.relu(video)
        out = video
        out = out.squeeze(3)
        out = out.squeeze(3)

        out = out.mean(2)
        
        
        #out=self.sofm(out)
        #c=self.sofm(c)
        out_logits=self.linearflu((torch.cat([c, out], dim=-1)))
        #out_logits = out+c
        out_logits = self.sofm(out_logits)
        
        return  out_logits





