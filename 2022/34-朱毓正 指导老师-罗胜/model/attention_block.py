import torch
import math
from torch import nn

class ECAnet(nn.Module):
    def __init__(self,channel,gamma=2, b=1):
        super(ECAnet,self).__init__()
        kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        padding = kernel_size // 2

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1,1,kernel_size=kernel_size,padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view(b,1,c)
        out = self.conv(avg)
        out = self.sigmoid(out).view([b,c,1,1])

        return out * x




if __name__ == '__main__':
    model = ECAnet(23)
    print(model)
    x = torch.ones([2,23,30,1])
    output = model(x)