"""
@Name: ChannelAttention_class.py
@Auth: SniperIN_IKBear
@Date: 2022/11/30-20:39
@Desc: 
@Ver : 0.0.0
"""
from torch import nn

"""
Channnel attetion module(通道注意力模块)
"""
# 参考 https://github.com/luuuyi/CBAM.PyTorch
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        # self.fc1 = nn.Conv2d(in_planes, in_planes // 6, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        # self.fc2 = nn.Conv2d(in_planes // 6, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)
