import torch
import numpy as np
import math
import torch.nn as nn

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
class hcn(nn.Module):
    def __init__(self,inpointchannel,injointchanel,classnumber):
        super().__init__()
        self.hcn_point_S=hcn_point(inpointchannel)
        self.hcn_point_M = hcn_point(inpointchannel)
        self.hcn_cooccurrence_S = hcn_cooccurrence(injointchanel)
        self.hcn_cooccurrence_M = hcn_cooccurrence(injointchanel)
        self.conv5=nn.Conv2d(128,128,3,padding=1)
        self.conv6=nn.Conv2d(128,256,3,padding=1)
        self.layer = nn.MaxPool2d(2, stride=2)
        self.relu=nn.ReLU()
        self.fc7=nn.Linear(1024,256)
        self.fc8=nn.Linear(256,classnumber)
        #self.sofm=nn.Softmax()

        self.drop5 = nn.Dropout(p=0.5)
        self.drop6 = nn.Dropout(p=0.5)
        self.drop7 = nn.Dropout(p=0.5)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
    def forward(self,x):
        batch,pople,frame,joint,channel=x.shape
        x=x.view(-1,frame,joint,channel)
        x = x.permute(0, 3, 1, 2).contiguous()
        vel1 = x[:, :, :1] * 0
        vel2 = x[:, :, 1:] - x[:, :, :-1]
        m = torch.cat((vel1, vel2), dim=2)

        x=self.hcn_point_S(x)
        x=x.permute(0,3,2,1)
        x=self.hcn_cooccurrence_S(x)
        m = self.hcn_point_M(m)
        m = m.permute(0,3,2,1)
        m = self.hcn_cooccurrence_M(m)
        c=torch.cat((x,m),dim=1)
        c=self.conv5(c)
        c=self.layer(c)
        c=self.relu(c)
        c = self.drop5(c)
        c = self.conv6(c)

        c = self.layer(c)
        c = self.relu(c)
        c = c.view(batch, pople, -1)  # N M D
        c = c.max(dim=1)[0]  # N D
        c=self.drop6(c)
        c=self.fc7(c)
        c=self.relu(c)
        c=self.drop7(c)
        c=self.fc8(c)
        #c=self.sofm(c)

        return c




