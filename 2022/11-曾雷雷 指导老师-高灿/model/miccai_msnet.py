import os, sys

sys.path.append('/data2/zengleilei/code/medical_code/rework/MSNet-main')
import torch
import torch.nn as nn
import torch.nn.functional as F
# from
# from model.res2net import res2net50_v1b_26w_4s
from model.res2net import *
import torchvision


class MSNet(nn.Module):
    # res2net based encoder decoder
    def __init__(self):
        super(MSNet, self).__init__()
        # ---- ResNet Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True))

        self.x5_x4_x3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                         nn.ReLU(inplace=True))
        self.x5_dem_4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                            nn.ReLU(inplace=True))

        self.level3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.x5_dem_5 = nn.Sequential(nn.Conv2d(2048, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                      nn.ReLU(inplace=True))

        self.output4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))
        self.output2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                     nn.ReLU(inplace=True))

        self.output4_b = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))
        self.output3_b = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))
        self.output2_b = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))
        self.output1_b = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64),
                                       nn.ReLU(inplace=True))
        self.output1 = nn.Sequential(nn.Conv2d(64, 1, kernel_size=3, padding=1))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        input = x

        # '''
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x1 = self.resnet.maxpool(x)  # bs, 64, 88, 88
        # ---- low-level features ----
        x2 = self.resnet.layer1(x1)  # bs, 256, 88, 88
        x3 = self.resnet.layer2(x2)  # bs, 512, 44, 44
        x4 = self.resnet.layer3(x3)  # bs, 1024, 22, 22
        x5 = self.resnet.layer4(x4)  # bs, 2048, 11, 11
        # '''

        x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        x5_4 = self.x5_x4(abs(F.upsample(x5_dem_1, size=x4.size()[2:], mode='bilinear') - x4_dem_1))
        x4_3 = self.x4_x3(abs(F.upsample(x4_dem_1, size=x3.size()[2:], mode='bilinear') - x3_dem_1))
        x3_2 = self.x3_x2(abs(F.upsample(x3_dem_1, size=x2.size()[2:], mode='bilinear') - x2_dem_1))
        x2_1 = self.x2_x1(abs(F.upsample(x2_dem_1, size=x1.size()[2:], mode='bilinear') - x1))

        x5_4_3 = self.x5_x4_x3(abs(F.upsample(x5_4, size=x4_3.size()[2:], mode='bilinear') - x4_3))
        x4_3_2 = self.x4_x3_x2(abs(F.upsample(x4_3, size=x3_2.size()[2:], mode='bilinear') - x3_2))
        x3_2_1 = self.x3_x2_x1(abs(F.upsample(x3_2, size=x2_1.size()[2:], mode='bilinear') - x2_1))

        x5_4_3_2 = self.x5_x4_x3_x2(abs(F.upsample(x5_4_3, size=x4_3_2.size()[2:], mode='bilinear') - x4_3_2))
        x4_3_2_1 = self.x4_x3_x2_x1(abs(F.upsample(x4_3_2, size=x3_2_1.size()[2:], mode='bilinear') - x3_2_1))

        x5_dem_4 = self.x5_dem_4(x5_4_3_2)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
            abs(F.upsample(x5_dem_4, size=x4_3_2_1.size()[2:], mode='bilinear') - x4_3_2_1))

        level4 = x5_4
        level3 = self.level3(x4_3 + x5_4_3)
        level2 = self.level2(x3_2 + x4_3_2 + x5_4_3_2)
        level1 = self.level1(x2_1 + x3_2_1 + x4_3_2_1 + x5_4_3_2_1)

        x5_dem_5 = self.x5_dem_5(x5)
        output4 = self.output4(F.upsample(x5_dem_5, size=level4.size()[2:], mode='bilinear') + level4)
        output3 = self.output3(F.upsample(output4, size=level3.size()[2:], mode='bilinear') + level3)
        output2 = self.output2(F.upsample(output3, size=level2.size()[2:], mode='bilinear') + level2)
        output1 = self.output1(F.upsample(output2, size=level1.size()[2:], mode='bilinear') + level1)

        output4_b = self.output4_b(
            torch.cat([self.softmax(F.upsample(x5_dem_5, size=level4.size()[2:], mode='bilinear')) * output4, output4],
                      dim=1))
        output3_b = self.output3_b(
            torch.cat([self.softmax(F.upsample(output4_b, size=level3.size()[2:], mode='bilinear')) * output3, output3],
                      dim=1))
        output2_b = self.output2_b(
            torch.cat([self.softmax(F.upsample(output3_b, size=level2.size()[2:], mode='bilinear')) * output2, output2],
                      dim=1))

        # output4_b = self.output4_b(torch.cat(
        #     [self.softmax(self.output4_b(F.upsample(x5_dem_5, size=level4.size()[2:], mode='bilinear'))) * output4,
        #      output4], dim=1))
        # output3_b = self.output3_b(torch.cat(
        #     [self.softmax(self.output3_b(F.upsample(output4_b, size=level3.size()[2:], mode='bilinear'))) * output3,
        #      output3], dim=1))
        # output2_b = self.output2_b(torch.cat(
        #     [self.softmax(self.output2_b(F.upsample(output3_b, size=level2.size()[2:], mode='bilinear'))) * output2,
        #      output2], dim=1))

        # output4 = output4 * output4_b
        # output3 = output3 * output3_b
        # output2 = output2 * output2_b

        output1_b = self.output1(F.upsample(output2_b, size=level1.size()[2:], mode='bilinear') + level1)  # 原来的代码
        # output1_b = self.output1(F.upsample(output2_b, size=level1.size()[2:], mode='bilinear') + output1)  # 原来的代码
        # output1 = self.output1(F.upsample(output1,size=level1.size()[2:], mode='bilinear') + level1)
        # output1_b = self.output1_b(torch.cat([self.softmax(self.output1(F.upsample(output2_b, size=level1.size()[2:], mode='bilinear'))) * output1,output1],dim=1))
        # output1_b = self.output2_b(torch.cat([self.softmax(F.upsample(output2_b, size=level1.size()[2:], mode='bilinear')) * output1,output1],dim=1))
        #
        output = F.upsample(output1_b, size=input.size()[2:], mode='bilinear')  # 原来的代码
        # output = F.upsample(output1, size=input.size()[2:], mode='bilinear')
        if self.training:
            return output
        return output


class LossNet(torch.nn.Module):
    def __init__(self, resize=True):
        super(LossNet, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        # blocks.append(torchvision.models.vgg16(pretrained=True).features[23:30].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.mean = torch.nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.std = torch.nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        # 设置权重
        loss_weight = [0.2, 0.2, 0.3, 0.3]
        flag = 0
        # 自己代码
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += loss_weight[flag] * torch.nn.functional.mse_loss(x, y)
            # loss +=  torch.nn.functional.mse_loss(x, y)  # 源代码
            flag += 1  # 自己加的代码
        return loss


if __name__ == '__main__':
    # lossnet = LossNet().cuda()
    # res = torchvision.models.resnet18()
    ras = MSNet().cuda()
    input_tensor = torch.randn(1, 3, 352, 352).cuda()
    # print(lossnet)
    print('*' * 20)
    # print(res)
    # out = ras(input_tensor)
    print(ras)
