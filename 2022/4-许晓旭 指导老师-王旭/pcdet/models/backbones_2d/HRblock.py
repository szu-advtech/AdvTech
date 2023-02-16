import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

BN_MOMENTUM = 0.01
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class HighResolutionBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HighResolutionBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.block = nn.ModuleList()
        for i in range(2):
            cur_layers = []
            for j in range(3):
                cur_layers.append(BasicBlock(in_channel[i], in_channel[i]))
            self.block.append(nn.Sequential(*cur_layers))

        self.fuse_layers = self._make_fuse_block()
        self.relu = nn.ReLU(inplace=True)
        self.transition_layer1 = nn.Sequential(
            nn.Conv2d(
                64, 128, 3, 2, 1, bias=False),
            nn.BatchNorm2d(
                128, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )
        self.transition_layer2 = nn.Sequential(
            nn.Conv2d(
                128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(
                256, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=True)
        )


    def _make_fuse_block(self):
        fuse_layers = []
        for i in range(2):
            fuse_layer = []
            for j in range(2):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(self.in_channel[j],
                                  self.out_channel[i],
                                  1,
                                  1,
                                  0,
                                  bias=False),
                        nn.BatchNorm2d(self.out_channel[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i-j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = self.out_channel[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.in_channel[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                            nn.BatchNorm2d(num_outchannels_conv3x3,
                                           momentum=BN_MOMENTUM)
                            ))
                        else:
                            num_outchannels_conv3x3 = self.out_channel[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(self.in_channel[j],
                                          num_outchannels_conv3x3,
                                          3, 2, 1, bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3,
                                               momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=True)
                            ))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)


    def forward(self, x):
        for i in range(2):
            x[i] = self.block[i](x[i])
        x_fuse = []
        for i in range(2):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, 2):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear')
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        y3 = self.transition_layer2(self.transition_layer1(x[0])) + \
                self.transition_layer2(x[1])
        x_fuse.append(y3)
        return x_fuse


if __name__ == '__main__':
    x1 = torch.randn((8, 64, 320, 320))
    x2 = torch.randn((8, 128, 160, 160))

    HR_block = HighResolutionBlock([64, 128], [64, 128])
    out = HR_block([x1, x2])
    for i in out:
        print(i.size())