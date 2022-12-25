import torch
import torch.nn as nn
import torch.nn.functional as F
from misc import torchutils

from net import resnet50


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # spotlight branch
        self.resnet50_spotlight = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1))

        self.spotlight_stage1 = nn.Sequential(
            self.resnet50_spotlight.conv1,
            self.resnet50_spotlight.bn1,
            self.resnet50_spotlight.relu,
            self.resnet50_spotlight.maxpool,
            self.resnet50_spotlight.layer1
        )

        self.spotlight_stage2 = nn.Sequential(
            self.resnet50_spotlight.layer2
        )

        self.spotlight_stage3 = nn.Sequential(
            self.resnet50_spotlight.layer3
        )

        self.spotlight_stage4 = nn.Sequential(
            self.resnet50_spotlight.layer4
        )

        self.spotlight_classifier = nn.Conv2d(2048, 20, 1, bias=False)
        self.spotlight_backbone = nn.ModuleList(
            [self.spotlight_stage1, self.spotlight_stage2, self.spotlight_stage3, self.spotlight_stage4])

        # compensation branch
        self.resnet50_compensation = resnet50.resnet50(pretrained=True, strides=(2, 2, 2, 1), use_amm=True)

        self.compensation_stage1 = nn.Sequential(
            self.resnet50_compensation.conv1,
            self.resnet50_compensation.bn1,
            self.resnet50_compensation.relu,
            self.resnet50_compensation.maxpool,
            self.resnet50_compensation.layer1
        )

        self.compensation_stage2 = nn.Sequential(
            self.resnet50_compensation.layer2
        )

        self.compensation_stage3 = nn.Sequential(
            self.resnet50_compensation.layer3
        )

        self.compensation_stage4 = nn.Sequential(
            self.resnet50_compensation.layer4
        )

        self.compensation_classifier = nn.Conv2d(2048, 20, 1, bias=False)
        self.compensation_backbone = nn.ModuleList(
            [self.compensation_stage1, self.compensation_stage2, self.compensation_stage3, self.compensation_stage4])

        # two branch classifier layer
        self.newly_added = nn.ModuleList([self.spotlight_classifier, self.compensation_classifier])

    def trainable_parameters(self):
        return (
            list(self.spotlight_backbone.parameters()), list(self.compensation_backbone.parameters()),
            list(self.newly_added.parameters()))

    def forward(self, x):
        x_copy = x.clone()

        # spotlight branch forward
        spotlight_x = self.spotlight_stage1(x)
        spotlight_x = self.spotlight_stage2(spotlight_x)
        spotlight_x = self.spotlight_stage3(spotlight_x)
        spotlight_x = self.spotlight_stage4(spotlight_x)

        # compute spotlight cam
        spotlight_cam = F.conv2d(spotlight_x, self.spotlight_classifier.weight)
        spotlight_cam = F.relu(spotlight_cam)
        spotlight_cam = spotlight_cam[0] + spotlight_cam[1].flip(-1)

        spotlight_x = torchutils.gap2d(spotlight_x, keepdims=True)
        spotlight_x = self.spotlight_classifier(spotlight_x)
        spotlight_x = spotlight_x.view(-1, 20)

        # compensation branch forward
        compensation_x = self.compensation_stage1(x_copy)
        compensation_x = self.compensation_stage2(compensation_x)
        compensation_x = self.compensation_stage3(compensation_x)
        compensation_x = self.compensation_stage4(compensation_x)

        # compute compensation cam
        compensation_cam = F.conv2d(compensation_x, self.compensation_classifier.weight)
        compensation_cam = F.relu(compensation_cam)
        compensation_cam = compensation_cam[0] + compensation_cam[1].flip(-1)

        compensation_x = torchutils.gap2d(compensation_x, keepdims=True)
        compensation_x = self.compensation_classifier(compensation_x)
        compensation_x = compensation_x.view(-1, 20)

        return spotlight_x, spotlight_cam, compensation_x, compensation_cam


class CAM(Net):

    def __init__(self, stride=16):
        super(CAM, self).__init__()

    def forward(self, x, step=1):
        x_copy = x.clone()

        # spotlight branch
        if step == 1:
            spotlight_x = self.spotlight_stage1(x)
            spotlight_x = self.spotlight_stage2(spotlight_x)
            spotlight_x = self.spotlight_stage3(spotlight_x)
            spotlight_x = self.spotlight_stage4(spotlight_x)

            # compute spotlight cam
            spotlight_cam = F.conv2d(spotlight_x, self.spotlight_classifier.weight)
            return spotlight_cam

        # compensation branch
        elif step == 2:
            compensation_x = self.compensation_stage1(x_copy)
            compensation_x = self.compensation_stage2(compensation_x)
            compensation_x = self.compensation_stage3(compensation_x)
            compensation_x = self.compensation_stage4(compensation_x)

            # compute compensation cam
            compensation_cam = F.conv2d(compensation_x, self.compensation_classifier.weight)
            return compensation_cam
