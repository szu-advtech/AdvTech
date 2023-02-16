"""
This code was based on the file resnet.py (https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py)
from the pytorch/vision library (https://github.com/pytorch/vision).

The original license is included below:

BSD 3-Clause License

Copyright (c) Soumith Chintala 2016,
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import torch.nn as nn
import torch
from path_index import *

__all__ = ['ResNet', 'resnet18']


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class BasicBlockFilm(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockFilm, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._film(out, gamma1, beta1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self._film(out, gamma2, beta2)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)  ############!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        return out

    def _film(self, x, gamma, beta):
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        return gamma * x + beta


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class BasicBlockFilm_senet(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockFilm_senet, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # self.leakyrelu = nn.LeakyReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SELayer(planes, 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x, gamma1, beta1, gamma2, beta2):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self._film(out, gamma1, beta1)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self._film(out, gamma2, beta2)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def _film(self, x, gamma, beta):
        gamma = gamma[None, :, None, None]
        beta = beta[None, :, None, None]
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers):
        super(ResNet, self).__init__()
        self.initial_pool = False
        inplanes = self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=metatrain_kernel_size, stride=2, padding=1, bias=False) #This kernel_size has been changed to 7(Orig.5). Notice!
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, inplanes, layers[0])
        self.layer2 = self._make_layer(block, inplanes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, inplanes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, inplanes * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    # def l2_norm(self,input):

    #     input_size = input.size()
    #     buffer = torch.pow(input, 2)

    #     normp = torch.sum(buffer, 1).add_(1e-10)
    #     norm = torch.sqrt(normp)

    #     _output = torch.div(input, norm.view(-1, 1).expand_as(input))

    #     output = _output.view(input_size)

    #     return output

    def forward(self, x, param_dict=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.l2_norm(x)

        return x

    def get_layer_output(self, x, param_dict, layer_to_return):
        if layer_to_return == 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            if self.initial_pool:
                x = self.maxpool(x)
            return x
        else:
            resnet_layers = [self.layer1, self.layer2, self.layer3, self.layer4]
            layer = layer_to_return - 1
            for block in range(self.layers[layer]):
                x = resnet_layers[layer][block](x, param_dict[layer][block]['gamma1'],
                                                param_dict[layer][block]['beta1'],
                                                param_dict[layer][block]['gamma2'], param_dict[layer][block]['beta2'])
            return x

    @property
    def output_size(self):
        return 512


class FilmResNet(ResNet):

    def __init__(self, block, layers):
        ResNet.__init__(self, block, layers)
        self.layers = layers

    def forward(self, x, param_dict):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.initial_pool:
            x = self.maxpool(x)

        for block in range(self.layers[0]):
            x = self.layer1[block](x, param_dict[0][block]['gamma1'], param_dict[0][block]['beta1'],
                                   param_dict[0][block]['gamma2'], param_dict[0][block]['beta2'])
        for block in range(self.layers[1]):
            x = self.layer2[block](x, param_dict[1][block]['gamma1'], param_dict[1][block]['beta1'],
                                   param_dict[1][block]['gamma2'], param_dict[1][block]['beta2'])
        for block in range(self.layers[2]):
            x = self.layer3[block](x, param_dict[2][block]['gamma1'], param_dict[2][block]['beta1'],
                                   param_dict[2][block]['gamma2'], param_dict[2][block]['beta2'])
        for block in range(self.layers[3]):
            x = self.layer4[block](x, param_dict[3][block]['gamma1'], param_dict[3][block]['beta1'],
                                   param_dict[3][block]['gamma2'], param_dict[3][block]['beta2'])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        # x = self.l2_norm(x)

        return x

def resnet10():
    model = ResNet(BasicBlock, [1, 1, 1, 1])
    return model




def resnet18(pretrained=False, pretrained_model_path=None, mt=False, **kwargs):
    if mt:
        return resnet18_mt(pretrained, pretrained_model_path, **kwargs)

    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path, map_location={'cuda:3': 'cuda:0'})
        new_dict = copyStateDict(ckpt_dict)
        keys = []
        for k, v in new_dict.items():
            if k.startswith('fc'):
                continue
            keys.append(k)

        new_dict = {k: new_dict[k] for k in keys}
        model.load_state_dict(new_dict)
    return model


def film_resnet18(pretrained=False, pretrained_model_path=None, mt=False, **kwargs):
    if mt:
        return film_resnet18_mt(pretrained, pretrained_model_path, **kwargs)

    model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], **kwargs)
    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path, map_location='cpu')
        new_dict = copyStateDict(ckpt_dict)
        keys = []
        for k, v in new_dict.items():
            if k.startswith('fc'):
                continue 
            keys.append(k)

        new_dict = {k: new_dict[k] for k in keys}
        model.load_state_dict(new_dict,strict=True)
    return model


from collections import OrderedDict


def copyStateDict(state_dict):
    if list(state_dict.keys())[0].startswith('module'):
        start_idx = 1
    else:
        start_idx = 0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = '.'.join(k.split('.')[start_idx:])
        new_state_dict[name] = v
    return new_state_dict


def film_resnet10(pretrained=False, pretrained_model_path=None, mt=False, **kwargs):
    """
        Constructs a FiLM adapted ResNet-10 model.
    """
    # nl = get_normalization_layer(batch_normalization)
    if mt:
        return film_resnet18_mt(pretrained, pretrained_model_path, **kwargs)

    model = FilmResNet(BasicBlockFilm, [1, 1, 1, 1], **kwargs)

    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        new_dict = copyStateDict(ckpt_dict)
        keys = []
        for k, v in new_dict.items():
            if k.startswith('fc'):
                continue

            keys.append(k)
        new_dict = {k: new_dict[k] for k in keys}
        model.load_state_dict(new_dict)

    return model


def film_resnet34(pretrained=False, pretrained_model_path=None, mt=False, **kwargs):
    """
        Constructs a FiLM adapted ResNet-10 model.
    """
    # nl = get_normalization_layer(batch_normalization)
    if mt:
        return film_resnet18_mt(pretrained, pretrained_model_path, **kwargs)

    model = FilmResNet(BasicBlockFilm, [3, 4, 6, 3], **kwargs)

    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        new_dict = copyStateDict(ckpt_dict)
        keys = []
        for k, v in new_dict.items():
            if k.startswith('fc'):
                continue

            keys.append(k)

        new_dict = {k: new_dict[k] for k in keys}
        model.load_state_dict(new_dict)

    return model

def resnet34():
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    return model


def resnet18_mt(pretrained=False, pretrained_model_path=None, **kwargs):
    """
        Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict)
    return model


def film_resnet18_mt(pretrained=False, pretrained_model_path=None, **kwargs):
    model = FilmResNet(BasicBlockFilm, [2, 2, 2, 2], **kwargs)
    model_state = model.state_dict()
    if pretrained:
        ckpt_dict = torch.load(pretrained_model_path)
        model.load_state_dict(ckpt_dict['resnet_dict'])
    return model
