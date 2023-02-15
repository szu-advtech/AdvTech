import torch
import torch.nn.functional as F
import torch.nn as nn

from model.network.BaseModel import BaseNetwork
from model.network.architecture import ResidualBlock
from model.network.architecture import SPADEResnetBlock
from model.network.normalizition import get_nospade_norm_layer

class U_Net(BaseNetwork):
    
    def __init__(self,input_channels = 3):
        super().__init__()
        kw = 4
        pw = int((kw -1)//2)
        nf = 64
        self.input_channels = input_channels

        norm_layer = get_nospade_norm_layer('spectralinstance') #return a function --first SN,and then IN
        self.layer1 = norm_layer(nn.Conv2d(self.input_channels,nf,3,stride=1,padding=pw)) #3 - 128
        self.layer2 = nn.Sequential(
            norm_layer(nn.Conv2d(nf *1,nf*2,3,stride=1,padding=1)),    # 128 - 64
            ResidualBlock(nf*2,nf*2)
        )
        self.layer3 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*2,nf*4,kw,stride=2,padding=pw)),     #64- 32
            ResidualBlock(nf*4,nf*4)
        )
        self.layer4 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*4,nf*4,kw,stride=2,padding=pw)),     #32 - 16
            ResidualBlock(nf*4,nf*4)
        )
        self.layer5 = nn.Sequential(
            norm_layer(nn.Conv2d(nf*4,nf*4,kw,stride=2,padding=pw)),     #16 - 8
            ResidualBlock(nf*4,nf*4)
        )
        self.head_0 = SPADEResnetBlock(nf*4,nf*4,ic=20)
        self.G_middle_0 = SPADEResnetBlock(nf*4,nf*4,ic = 20)
        self.G_middle_1 = SPADEResnetBlock(nf*4,nf*2,ic = 20)
        self.G_middle_2 = SPADEResnetBlock(nf*2,nf,ic = 20)
        self.up = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
    
    def forward(self,input,seg):
        #128 - 128   c 3-64
        x1 = self.layer1(input)
        #128 - 128   c 64-128
        x2 = self.layer2(self.actvn(x1))
        #128 - 64    c 128-256
        x3 = self.layer3(self.actvn(x2))
        #64 - 32     c 256-256
        x4 = self.layer4(self.actvn(x3))
        #32 - 16     c 256-256
        x5 = self.layer5(self.actvn(x4))
        #16 - 16 bottomleck c 256-256
        x6 = self.head_0(x5,seg)
        #16 - 32 c 256-256
        x7 = self.G_middle_0(self.up(x6) + x4 ,  seg)
        #32 - 64 
        x8 = self.G_middle_1(self.up(x7) + x3,   seg)
        #64 - 128
        x9 = self.G_middle_2(self.up(x8) + x2,   seg)
        return [x6,x7,x8,x9]
        pass
    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)