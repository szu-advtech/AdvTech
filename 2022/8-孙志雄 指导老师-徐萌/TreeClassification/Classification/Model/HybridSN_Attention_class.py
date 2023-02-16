"""
@Name: HybridSN_Attention_class.py
@Auth: SniperIN_IKBear
@Date: 2022/11/30-20:42
@Desc: 
@Ver : 0.0.0
"""
import torch

from .ChannelAttention_class import ChannelAttention
from .SpatialAttention_class import SpatialAttention

"""
加上注意力机制的HybridSN模型如下：
"""
from torch import nn

class HybridSN_Attention(nn.Module):
    class_num = 8
    def __init__(self, in_channels=1,out_out_channels=class_num):
        super(HybridSN_Attention, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(7, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.ReLU()
        )
        # 通道和空间注意力
        self.ca = ChannelAttention(32 * 18)
        self.sa = SpatialAttention()

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=32 * 18, out_channels=64, kernel_size=(3, 3)),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 17 * 17, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, out_out_channels)
        )

    def forward(self, x):
        x = self.conv3d_features(x)
        x = x.view(x.size()[0], x.size()[1] * x.size()[2], x.size()[3], x.size()[4])

        x = self.ca(x) * x
        x = self.sa(x) * x

        x = self.conv2d_features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return x

def test():# # #随机输入，测试网络结构是否通
    x = torch.randn(1, 1, 30, 25, 25)
    net_attention = HybridSN_Attention()
    y = net_attention(x)
    print(y.shape)