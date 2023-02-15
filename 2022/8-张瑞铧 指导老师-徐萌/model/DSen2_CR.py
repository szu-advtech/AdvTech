import torch
import torch.nn as nn

# ResBlock模块
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, F=256, res_scale=0.1):
        super(ResBlock, self).__init__()
        self.res_scale = res_scale

        self.res_block = nn.Sequential(
            nn.Conv2d(in_channels=F,out_channels=F,kernel_size=3,bias=False,stride=1,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=F,out_channels=F,kernel_size=3,bias=False,stride=1,padding=1)
        )


    def forward(self, x):
        out = self.res_block(x)
        return out * self.res_scale + x


# DSen2-CR模型
class DSen2_CR(nn.Module):
    def __init__(self, F=256, B=16, res_scale=0.1):
        super(DSen2_CR, self).__init__()
        backbone = [
            nn.Conv2d(in_channels=13, out_channels=F, kernel_size=3, bias=True, stride=1, padding=1),
            nn.ReLU()
        ]

        for i in range(B):
            backbone.append(ResBlock(in_channels=F, out_channels=F, res_scale=res_scale))

        backbone.append(nn.Conv2d(in_channels=F, out_channels=13, kernel_size=3, bias=True, stride=1, padding=1))

        self.backbone = nn.Sequential(*backbone)

    def forward(self, x):
        #return x[:,2:,:,:] + self.backbone(x)
        return x + self.backbone(x)
