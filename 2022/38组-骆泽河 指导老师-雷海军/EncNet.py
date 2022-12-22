import torch
import torch.nn as nn
import torch.nn.functional as F

from model import Encoding
from rest50 import ResNet


class EncModule(nn.Module):
    def __init__(self, in_channels, num_codes):
        super(EncModule, self).__init__()
        self.encoding_project = nn.Conv2d(
            in_channels,
            in_channels,
            1,
        )
        # TODO: resolve this hack
        # change to 1d
        self.encoding = nn.Sequential(
            Encoding(channels=in_channels, num_codes=num_codes),
            nn.BatchNorm1d(num_codes),
            nn.ReLU(inplace=True))
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels), nn.Sigmoid())

    def forward(self, x):
        """Forward function."""
        encoding_projection = self.encoding_project(x)
        encoding_feat = self.encoding(encoding_projection).mean(dim=1)

        # print("encoding_feat2: ",encoding_feat.shape)

        batch_size, channels, _, _ = x.size()
        gamma = self.fc(encoding_feat)
        y = gamma.view(batch_size, channels, 1, 1)
        output = F.relu_(x + x * y)
        return encoding_feat, output


class EncHead(nn.Module):

    def __init__(self, num_classes=33,
                 num_codes=32,
                 use_se_loss=True,
                 add_lateral=False,
                 **kwargs):
        super(EncHead, self).__init__()
        self.use_se_loss = use_se_loss
        self.add_lateral = add_lateral
        self.num_codes = num_codes
        self.in_channels = [256, 512, 1024, 2048]
        self.channels = 512
        self.num_classes = num_classes
        self.bottleneck = nn.Conv2d(
            self.in_channels[-1],
            self.channels,
            3,
            padding=1,
        )

        if add_lateral:
            self.lateral_convs = nn.ModuleList()
            for in_channels in self.in_channels[:-1]:  # skip the last one
                self.lateral_convs.append(
                    nn.Conv2d(
                        in_channels,
                        self.channels,
                        1,
                    ))
            self.fusion = nn.Conv2d(
                len(self.in_channels) * self.channels,
                self.channels,
                3,
                padding=1,
            )

        self.enc_module = EncModule(
            self.channels,
            num_codes=num_codes,
        )
        self.cls_seg = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 33, 3, padding=1)
        )

        if self.use_se_loss:
            self.se_layer = nn.Linear(self.channels, self.num_classes)

    def forward(self, inputs):
        """Forward function."""
        feat = self.bottleneck(inputs[-1])
        if self.add_lateral:
            laterals = [
                nn.functional.interpolate(input=lateral_conv(inputs[i]), size=feat.shape[2:],
                                          mode='bilinear')
                for i, lateral_conv in enumerate(self.lateral_convs)
            ]
            feat = self.fusion(torch.cat([feat, *laterals], 1))
        encode_feat, output = self.enc_module(feat)
        output = nn.functional.interpolate(input=output, scale_factor=8, mode="bilinear")
        output = self.cls_seg(output)
        if self.use_se_loss:
            se_output = self.se_layer(encode_feat)
            return output, se_output
        else:
            return output


class ENCNet(nn.Module):
    def __init__(self, num_classes):
        super(ENCNet, self).__init__()
        self.num_classes = num_classes
        self.backbone = ResNet.resnet50()
        self.decoder = EncHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.decoder(x)
        return x