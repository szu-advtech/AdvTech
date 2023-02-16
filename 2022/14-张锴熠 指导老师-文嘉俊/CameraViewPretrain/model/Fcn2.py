from torch import nn
import torch


class Fcn2(nn.Module):
    def __init__(self):
        super(Fcn2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=(5, 5), stride=1, padding=(2, 2)),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 5), stride=1, padding=(2, 2)),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 5), stride=1, padding=(2, 2)),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(5, 5), stride=1, padding=(2, 2)),
            nn.ReLU()
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), stride=1, padding=(2, 2)),
            nn.ReLU()
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), stride=1, padding=(2, 2)),
            nn.ReLU()
        )
        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(5, 5), stride=1, padding=(2, 2)),
            nn.ReLU()
        )
        self.max_pooling = nn.MaxPool2d((2, 2), stride=2)

    def forward(self, v2):
        out2 = self.conv1(v2)
        out2 = self.conv2(out2)
        out2 = self.max_pooling(out2)
        out2 = self.conv3(out2)
        out2 = self.conv4(out2)
        out2 = self.max_pooling(out2)
        out2 = self.conv5(out2)
        out2 = self.conv6(out2)
        out2 = self.conv7(out2)
        return out2
