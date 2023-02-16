from .resnet import *
import torch.nn as nn
import torch

class ResNetSeries(nn.Module):
    def __init__(self, pretrained):
        super(ResNetSeries, self).__init__()

        if pretrained == 'supervised':
            print(f'Loading supervised pretrained parameters!')
            self.resnet50 = resnet50(pretrained=True, use_amm=False)
            self.resnet50_2 = resnet50(pretrained=True, use_amm=True)

        elif pretrained == 'mocov2':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            self.resnet50 = resnet50(pretrained=True, use_amm=False)
            checkpoint = torch.load('moco_r50_v2-e3b0c442.pth', map_location="cpu")
            self.resnet50.load_state_dict(checkpoint['state_dict'], strict=False)

            self.resnet50_2 = resnet50(pretrained=True, use_amm=True)
            checkpoint = torch.load('moco_r50_v2-e3b0c442.pth', map_location="cpu")
            self.resnet50_2.load_state_dict(checkpoint['state_dict'], strict=False)

        elif pretrained == 'detco':
            print(f'Loading unsupervised {pretrained} pretrained parameters!')
            self.resnet50 = resnet50(pretrained=True, use_amm=False)
            checkpoint = torch.load('detco_200ep.pth', map_location="cpu")
            self.resnet50.load_state_dict(checkpoint['state_dict'], strict=False)

            self.resnet50_2 = resnet50(pretrained=True, use_amm=True)
            checkpoint = torch.load('detco_200ep.pth', map_location="cpu")
            self.resnet50_2.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            raise NotImplementedError

        self.conv1 = self.resnet50.conv1
        self.bn1 = self.resnet50.bn1
        self.relu = self.resnet50.relu
        self.maxpool = self.resnet50.maxpool
        # ========================================
        self.stage1 = nn.Sequential(self.resnet50.layer1)
        self.stage2 = nn.Sequential(self.resnet50.layer2)
        self.stage3 = nn.Sequential(self.resnet50.layer3)
        self.stage4 = nn.Sequential(self.resnet50.layer4)

        # ========================================
        self.stage2_1 = nn.Sequential(self.resnet50_2.layer1)
        self.stage2_2 = nn.Sequential(self.resnet50_2.layer2)
        self.stage2_3 = nn.Sequential(self.resnet50_2.layer3)
        self.stage2_4 = nn.Sequential(self.resnet50_2.layer4)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        tmp = x

        x1 = self.stage4(x)

        x2 = self.stage2_4(tmp)

        return torch.cat([x2, x1, x], dim=1)

class Disentangler(nn.Module):
    def __init__(self, cin):
        super(Disentangler, self).__init__()

        self.activation_head = nn.Conv2d(cin, 1, kernel_size=3, padding=1, bias=False)
        self.bn_head = nn.BatchNorm2d(1)

    def forward(self, x, inference=False):
        N, C, H, W = x.size()
        if inference:
            ccam = self.bn_head(self.activation_head(x))
        else:
            ccam = torch.sigmoid(self.bn_head(self.activation_head(x)))

        ccam_ = ccam.reshape(N, 1, H * W)                          # [N, 1, H*W]
        x = x.reshape(N, C, H * W).permute(0, 2, 1).contiguous()   # [N, H*W, C]
        fg_feats = torch.matmul(ccam_, x) / (H * W)                # [N, 1, C]
        bg_feats = torch.matmul(1 - ccam_, x) / (H * W)            # [N, 1, C]

        return fg_feats.reshape(x.size(0), -1), bg_feats.reshape(x.size(0), -1), ccam

class Network(nn.Module):
    def __init__(self, pretrained='mocov2', cin=None):
        super(Network, self).__init__()

        self.backbone = ResNetSeries(pretrained=pretrained)
        self.ac_head = Disentangler(cin)
        self.from_scratch_layers = [self.ac_head]

    def forward(self, x, inference=False):

        feats = self.backbone(x)
        # print('feats_shape is {}'.format(feats.shape))
        fg_feats, bg_feats, ccam = self.ac_head(feats, inference=inference)

        return fg_feats, bg_feats, ccam

    def get_parameter_groups(self):
        groups = ([], [], [], [])
        print('======================================================')
        for m in self.modules():
            if (isinstance(m, nn.Conv2d) or isinstance(m, nn.modules.normalization.GroupNorm)):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)
        return groups

def get_model(pretrained, cin=4096+1024):
    return Network(pretrained=pretrained, cin=cin)
