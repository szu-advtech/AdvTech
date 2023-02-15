import math

from channel import channel_selection, channel_change, reconstruction_errors
import torch
import torch.nn as nn
from collections import OrderedDict

norm_mean, norm_var = 0.0, 1.0

defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]
convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]


# 制造神经网络的结构
def make_layers(cfg):
    layers = nn.Sequential()
    in_channels = 3
    cnt = 0
    for i, v in enumerate(cfg):
        if v == 'M':
            layers.add_module('pool%d' % i, nn.MaxPool2d(kernel_size=2, stride=2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            cnt += 1

            layers.add_module('conv%d' % i, conv2d)
            layers.add_module('norm%d' % i, nn.BatchNorm2d(v))
            layers.add_module('relu%d' % i, nn.ReLU(inplace=True))
            in_channels = v

    return layers


class Thinet_VGG(nn.Module):
    def __init__(self, num_classes=10, init_weights=True, cfg=None):
        super(Thinet_VGG, self).__init__()
        self.features = nn.Sequential()

        # Thinet
        self.named_modules_idx_list = dict()
        self.named_modules_list = dict()
        self.named_conv_list = dict()
        self.named_conv_idx_list = dict()
        self.original_conv_output = dict()
        self.stayed_channels = dict()

        if cfg is None:
            cfg = defaultcfg

        self.relucfg = relucfg
        self.covcfg = convcfg
        self.features = make_layers(cfg[:-1])
        self.classifier = nn.Sequential(OrderedDict([
            ('linear1', nn.Linear(cfg[-2], cfg[-1])),
            ('norm1', nn.BatchNorm1d(cfg[-1])),
            ('relu1', nn.ReLU(inplace=True)),
            ('linear2', nn.Linear(cfg[-1], num_classes)),
        ]))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        # 提取特征，也就是计算
        x = self.features(x, )

        x = nn.AvgPool2d(2)(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    # 用Parameter得到卷积层
    def get(self):
        res = []
        cnt = 0
        for name, param in self.named_parameters():
            if name.startswith("features.conv") and name.endswith("weight"):
                res.append(cnt)
            cnt += 1
        return res

    # 所需类的字典
    def init_Thi(self):
        i = 0
        for idx, m in enumerate(self.features):
            if isinstance(m, torch.nn.Conv2d):
                self.named_modules_idx_list['{}.conv'.format(i)] = idx
                self.named_modules_list['{}.conv'.format(i)] = m
                self.named_conv_idx_list['{}.conv'.format(i)] = idx
                self.named_conv_list['{}.conv'.format(i)] = m
            elif isinstance(m, torch.nn.BatchNorm2d):
                self.named_modules_idx_list['{}.bn'.format(i)] = idx
                self.named_modules_list['{}.bn'.format(i)] = m
                i += 1
        # print(self.named_modules_idx_list)
        # print(self.named_modules_list)
        # print(self.named_conv_idx_list)
        # print(self.named_conv_list)

    # 记录选定数据集每层的输出
    def record_conv_output(self, inputs, device):
        x = inputs.to(device)
        i = 0
        for m in self.features:
            x = m(x)
            if isinstance(m, torch.nn.Conv2d):
                self.original_conv_output['{}.conv'.format(i)] = x.data
                i += 1

    # 剪枝后训练
    def my_train(self, epoch, trainloader, criterion, optimizer, device):
        for i in range(epoch):
            all_loss, total, correct = 0, 0, 0
            for _, (inputs, targets) in enumerate(trainloader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                # 清空过往梯度
                optimizer.zero_grad()
                outputs = self.forward(inputs)
                outputs = outputs.to(device)
                loss = criterion(outputs, targets)

                all_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 计算损失
                loss.backward()

                # 更新模型
                optimizer.step()

    # 进行剪枝
    def compress(self, inputs, compressrate, trainloader, criterion, optimizer, device):
        # 最小化重构误差需要原本每层的输出
        self.record_conv_output(inputs, device)
        inputs = inputs.to(device)
        for i, module in enumerate(list(self.named_conv_list.values())[:-1]):
            print('Pruning %d Conv Layer' % (i+1))
            # 下一个卷积层
            next_module_idx = self.named_modules_idx_list[str(i + 1) + '.conv']
            # 下一个卷积层的module，为了filter
            next_module = self.named_modules_list[str(i + 1) + '.conv']
            # 下一层卷积的输入
            next_input_features = self.features[:next_module_idx](inputs)
            # 蛮力法选择要裁剪的通道
            indices_stayed, indices_pruned = channel_selection(next_input_features, next_module, device, compressrate)
            # print(indices_stayed)
            # 保存结果
            self.stayed_channels[str(i) + '.conv'] = set(indices_stayed)
            # 当前卷积对应的BatchNorm2d
            bn = self.named_modules_list[str(i) + '.bn']
            # 根据前面得到的通道，修剪当前层的filter组，与下一层的filter组内的相应通道
            channel_change(module, bn, next_module, indices_stayed)

            # 原本第i层卷积的输出
            next_original_output_features = self.original_conv_output[str(i + 1) + '.conv']
            # 现在第i层卷积的输出
            next_conv_idx = self.named_conv_idx_list[str(i + 1) + '.conv']
            next_pruned_inputs_features = self.features[:next_conv_idx](inputs)
            # 最小化重构误差
            reconstruction_errors(next_module, next_pruned_inputs_features, next_original_output_features)
            self.my_train(2, trainloader, criterion, optimizer, device)
        self.my_train(10, trainloader, criterion, optimizer, device)
