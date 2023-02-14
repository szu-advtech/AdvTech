# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial

from timm.models.vision_transformer import VisionTransformer, _cfg, PatchEmbed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import numpy as np
import torchvision
import matplotlib.pyplot as plt
from backbone import VGG
from backbone import resnet


# class SAFA(VGG.VGG):
class SAFA(resnet.ResNet):
    def __init__(self, crop=False, save=None, img_size=(224, 224), *args, **kwargs):
        # super().__init__(VGG.vgg16_bn())
        super().__init__(resnet.BasicBlock, [3, 4, 6, 3])
        self.max_pooling_along_channels = nn.AdaptiveAvgPool3d((1, None, None))
        self.spatial_aware_importance_generator = nn.ModuleList()

        height = img_size[0]//32
        width = img_size[1]//32
        N = height * width
        for _ in range(4):
            self.spatial_aware_importance_generator.append(
                nn.Sequential(
                    nn.Linear(N, N),
                    nn.ReLU(inplace=True),
                    nn.Linear(N, N),
                    nn.Sigmoid(),
                )
            )
        self.head = nn.Linear(N * 512 * 4, 1000)


    def forward(self, x, atten=None, indexes=None):

        output = self.conv1(x)
        output = self.maxpool(output)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        # output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)

        # x = self.features(x)
        x = output
        x_dist = self.max_pooling_along_channels(x)
        x_dist = x_dist.reshape(x.shape[0], x.shape[2] * x.shape[3])
        x_dist = self.spatial_aware_importance_generator[0](x_dist)
        x_dist = x_dist.reshape(x_dist.shape[0], x.shape[2], x.shape[3])
        x_dist = x_dist.view(x_dist.shape[0], 1, x_dist.shape[1], x_dist.shape[2])
        x_dist = x * x_dist
        x_dist = x_dist.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        for i in range(1, 4):
            x_mid = self.max_pooling_along_channels(x)
            x_mid = x_mid.reshape(x.shape[0], x.shape[2] * x.shape[3])
            x_mid = self.spatial_aware_importance_generator[i](x_mid)
            x_mid = x_mid.reshape(x.shape[0], x.shape[2], x.shape[3])
            x_mid = x_mid.view(x_mid.shape[0], 1, x_mid.shape[1], x_mid.shape[2])
            x_mid = x * x_mid
            x_mid = x_mid.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
            x_dist = torch.cat((x_dist, x_mid), dim=1)
        x = self.head(x_dist)
        return x

@register_model
def safa(pretrained=True, img_size=(224, 224), num_classes = 1000, **kwargs):
    model = SAFA(img_size=img_size, **kwargs)
    return model

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

if __name__ == '__main__':
    model = safa()
    print(model)

