import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import random
import os

myseed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
random.seed(myseed)
os.environ['PYTHONHASHSEED'] = str(myseed)
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)


# class Net(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()
#         self.model1 = nn.Sequential(
#             nn.Conv2d(3, 32, 5, padding=2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 32, 5, padding=2),
#             nn.MaxPool2d(2),
#             nn.Conv2d(32, 64, 5, padding=2),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(1024, 64),
#             nn.Linear(64, 10)
#         )
#
#     def forward(self, x):
#         x = self.model1(x)
#         return x

# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         # 96->64 64->128
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(2, 2), padding=1),
#             nn.Dropout(0.5)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(2, 2), padding=1),
#             nn.Dropout(0.5)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0),
#             nn.ReLU(),
#             nn.Conv2d(128, 10, kernel_size=(1, 1), padding=0),
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.squeeze(x)
#         return x


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(5, 5), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=2),
#             nn.Dropout(0.5)
#         )
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(5, 5), padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=(3, 3), stride=2),
#             nn.Dropout(0.5)
#         )
#         self.conv3 = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
#             nn.ReLU(),
#             nn.Conv2d(128, 128, kernel_size=(1, 1), padding=0),
#             nn.ReLU(),
#             nn.Conv2d(128, 10, kernel_size=(1, 1), padding=0),
#         )
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.conv2(x)
#         x = self.conv3(x)
#         x = F.adaptive_avg_pool2d(x, (1, 1))
#         x = torch.squeeze(x)
#         return x

class Net(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(in_features=1024, out_features=256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=10)
        )

    def forward(self, x):
        x = self.model1(x)
        return x


class ALL_CNN(nn.Module):
    def __init__(self):
        super(ALL_CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), padding=1),
            nn.RReLU(),
            nn.Conv2d(96, 96, kernel_size=(3, 3), padding=1),
            nn.RReLU(),
            nn.Conv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Dropout(0.5)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), padding=1),
            nn.RReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), padding=1),
            nn.RReLU(),
            nn.Conv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.Dropout(0.5)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), padding=1),
            nn.RReLU(),
            nn.Conv2d(192, 192, kernel_size=(1, 1), padding=0),
            nn.RReLU(),
            nn.Conv2d(192, 10, kernel_size=(1, 1), padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.squeeze(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18(num_classes=10):

    return ResNet(ResidualBlock, num_classes=num_classes)

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )