"""
@Name: HybridSN_BN_Attention.py
@Auth: SniperIN_IKBear
@Date: 2022/11/30-20:43
@Desc: 
@Ver : 0.0.0
"""
from torch import nn

from HybridSN.Model.ChannelAttention_class import ChannelAttention
from HybridSN.Model.SpatialAttention_class import SpatialAttention

"""
加上Batch Normalization、注意力机制的HybridSN模型如下：
"""


class HybridSN_BN_Attention(nn.Module):
    def __init__(self, in_channels=1, out_channels=class_num):
        super(HybridSN_BN_Attention, self).__init__()
        self.conv3d_features = nn.Sequential(
            nn.Conv3d(in_channels, out_channels=8, kernel_size=(7, 3, 3)),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(5, 3, 3)),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        self.ca = ChannelAttention(32 * 18)
        self.sa = SpatialAttention()

        self.conv2d_features = nn.Sequential(
            nn.Conv2d(in_channels=32 * 18, out_channels=64, kernel_size=(3, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 17 * 17, 256),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(128, 16)
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
