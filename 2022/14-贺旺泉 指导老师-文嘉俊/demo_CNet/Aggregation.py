
import torch
from torch import nn


class AggregationModule(nn.Module):
    """function of Aggregation Contextual features."""
    def __init__(self,int_channel,out_channel,kerner_size):
        super(AggregationModule, self).__init__()
        self.int_channel = int_channel
        self.out_channel = out_channel
        padding = kerner_size // 2

        self.reduce_conv = nn.Conv2d(int_channel, out_channel, kernel_size=3, padding=1,
                                     bias=True)
        self.t1 = nn.Conv2d(int_channel, out_channel, kernel_size=(kerner_size, 1), padding=(padding, 0),
                                     groups=out_channel, bias=True)
        self.t2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kerner_size), padding=(0, padding),
                                     groups=out_channel, bias=True)
        self.p1 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kerner_size), padding=(0, padding),
                                     groups=out_channel, bias=True)
        self.p2 = nn.Conv2d(out_channel, out_channel, kernel_size=(kerner_size, 1), padding=(padding, 0),
                                     groups=out_channel, bias=True)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.relu(self.norm(self.reduce_conv(x)))

        x1 = self.t1(x)
        x1 = self.t2(x1)

        x2 = self.p1(x)
        x2 = self.p1(x2)

        out = self.relu(self.norm(x1 + x2))
        return out

class AggregationModule_Group(nn.Module):
    """function of Aggregation Contextual features."""
    def __init__(self,int_channel,out_channel,kerner_size):
        super(AggregationModule_Group, self).__init__()
        self.int_channel = int_channel
        self.out_channel = out_channel
        # padding = kerner_size // 2
        #
        # self.reduce_conv = nn.Conv2d(int_channel, out_channel, kernel_size=3, padding=1,
        #                              bias=True)
        # self.t1 = nn.Conv2d(int_channel, out_channel, kernel_size=(kerner_size, 1), padding=(padding, 0),
        #                              groups=out_channel, bias=True)
        # self.t2 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kerner_size), padding=(0, padding),
        #                              groups=out_channel, bias=True)
        # self.p1 = nn.Conv2d(out_channel, out_channel, kernel_size=(1, kerner_size), padding=(0, padding),
        #                              groups=out_channel, bias=True)
        # self.p2 = nn.Conv2d(out_channel, out_channel, kernel_size=(kerner_size, 1), padding=(0, padding),
        #                              groups=out_channel, bias=True)
        self.relu = nn.ReLU()
        self.norm = nn.BatchNorm2d(out_channel)

        # Local
        self.conv_l1 = nn.Conv2d(int_channel, out_channel, kernel_size=1, padding=0, stride=1, bias=True)
        self.sigmoid_l1 = nn.Sigmoid()
        # X
        self.avg_l2 = nn.AdaptiveAvgPool2d((None, 1))
        self.conv_l2 = nn.Conv2d(int_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.sigmoid_l2 = nn.Sigmoid()

        # Y
        self.avg_l3 = nn.AdaptiveAvgPool2d((1, None))
        self.conv_l3 = nn.Conv2d(int_channel, out_channel, kernel_size=1, stride=1, padding=0)
        self.sigmoid_l3 = nn.Sigmoid()

        # Global
        self.ave_l4 = nn.AdaptiveAvgPool2d(1)
        self.conv_l4 = nn.Linear(int_channel, out_channel)
        self.sigmoid_l4 = nn.Sigmoid()

    def forward(self, x):
        #local
        x1 = self.conv_l1(x)
        x1 = self.sigmoid_l1(x1)
        x11 = x * x1
        # X
        # x2 = self.avg_l2(x)
        # x2 = self.conv_l2(x2)
        # x2 = self.sigmoid_l2(x2)
        # x22 = x * x2.expand_as(x)
        #
        # # Y
        # x3 = self.avg_l3(x)
        # x3 = self.conv_l3(x3)
        # x3 = self.sigmoid_l3(x3)
        # x33 = x * x3.expand_as(x)

        #Global
        bn, c, h, w = x.size()
        x4 = self.ave_l4(x).view(bn, c)
        x4 = self.conv_l4(x4)
        x4 = self.sigmoid_l4(x4).view(bn, c, 1, 1)
        x44 = x * x4.expand_as(x)

        # out = self.relu(self.norm(x11 + x22 + x33 + x44))
        out = self.relu(self.norm(x11 + x44))
        return out