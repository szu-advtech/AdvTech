# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from torchvision.models import ResNet


def get_se_resnet_50(path_state_dict, device):

    model = se_resnet50(pretrained=False)
    if path_state_dict:
        state_dict = torch.load(path_state_dict)
        model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    return model

# 论文核心 SE Block, 这里称为 SE layer
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # 这里是squeeze操作，作全局平局池化，将通道之间的信息进行融合
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            # Linear: from channel to channel/16
            nn.Linear(channel, channel // reduction, bias=False),
            # ReLu：进行一次激活函数
            nn.ReLU(inplace=True),
            # Linear: from channel/16 to channel
            nn.Linear(channel // reduction, channel, bias=False),
            # Sigmoid: 激活到0-1，代表每个通道的重要性
            nn.Sigmoid()
        )


    def forward(self, x):
        batch_size, channel, _, _ = x.size()
        # avg_pool(x).squeeze().shape # squeeze()函数也可以将所有通道个数为1的进行挤压
        # x通过AdaptiveAvgPool2d(1)以后将得到（batch size, channel, 1, 1）, 然后view（b,c）意思是按照b,c进行展开
        y = self.avg_pool(x).view(batch_size, channel)
        # 现在y得到的是每一个通道对应的分数（0-1），然后需要将其与通道内容相乘，
        # 具体操作使用到了tensor的内置函数expand_as(把一个tensor变成和函数括号内一样形状的tensor，用法与expand类似，相当于expand(tensor.size())
        # x是【batch_size, channel,H,W】形状的特征图，y是【batch_size, channel,1,1】大小的channel-wise分数，然后需要将其相乘
        y = self.fc(y).view(batch_size, channel, 1, 1)
        return x * y.expand_as(x)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        # 参数列表里的 * 星号，标志着位置参数的就此终结，之后的那些参数，都只能以关键字形式来指定。
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        # 参数列表里的 * 星号，标志着位置参数的就此终结，之后的那些参数，都只能以关键字形式来指定。
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




def se_resnet50(num_classes=1_000, pretrained=False):

    model = ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


# 在resetnet中的block插入senet模块
class CifarSEBasicBlock(nn.Module):

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, reduction=16):
        super(CifarSEBasicBlock, self).__init__()
        #
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        # ReLU: 激活层， inplace进行原地操作，节省内存
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, reduction)
        if inplanes != planes:
            # downsample，作用是改变identity的通道数，使之与out的通道数相匹配。
            self.downsample = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                                            nn.BatchNorm2d(planes))
        else:
            self.downsample = lambda x: x
        self.stride = stride

    def forward(self, x):
        residual = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        out += residual
        out = self.relu(out)

        return out

# 这里是CifarSEResNet模块
class CifarSEResNet(nn.Module):
    # 其中block参决定是BasicBlock
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEResNet, self).__init__()
        self.inplane = 16
        self.conv1 = nn.Conv2d(
            3, self.inplane, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.relu = nn.ReLU(inplace=True)
        # layer1中stride=1, layer2-4中stride=2
        self.layer1 = self._make_layer(
            block, 16, blocks=n_size, stride=1, reduction=reduction)
        self.layer2 = self._make_layer(
            block, 32, blocks=n_size, stride=2, reduction=reduction)
        self.layer3 = self._make_layer(
            block, 64, blocks=n_size, stride=2, reduction=reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.initialize()

    def initialize(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # 这里是将若干个block块的集合
    def _make_layer(self, block, planes, blocks, stride, reduction):
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.inplane, planes, stride, reduction))
            self.inplane = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 三个layer层，每个layer层有blocks个块，每个blocks块有两个卷积层，所以总共的resnet有2*blocks*layers个resnet层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class CifarSEPreActResNet(CifarSEResNet):
    def __init__(self, block, n_size, num_classes=10, reduction=16):
        super(CifarSEPreActResNet, self).__init__(
            block, n_size, num_classes, reduction)
        self.bn1 = nn.BatchNorm2d(self.inplane)
        self.initialize()

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)


def se_resnet20(**kwargs):

    model = CifarSEResNet(CifarSEBasicBlock, 3, **kwargs)
    return model



def se_resnet56(**kwargs):

    model = CifarSEResNet(CifarSEBasicBlock, 9, **kwargs)
    return model
