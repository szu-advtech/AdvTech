import argparse
from collections import OrderedDict
import torch
import torchvision
import torchvision.transforms as transforms
from torch import optim, nn
from thop import profile

import get_flops
from model.Anti_vgg16 import Anti_VGG
from model.mask_vgg16 import Mask_VGG
from model.random_vgg16 import Random_VGG
from model.thinet_vgg16 import Thinet_VGG
from model.vgg16 import VGG
from tools import get_feature, test, HRank


# 判断输出是否为True、False
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


if __name__ == '__main__':

    # 变量
    parser = argparse.ArgumentParser(description='PyTorch HRank')
    parser.add_argument(
        '--get_feature',
        default=True,
        type=boolean_string,
        help='Compute Rank')
    parser.add_argument(
        '--compress_rate',
        type=float,
        default=0.2,
        help='Compress Rate')
    parser.add_argument(
        '--gpu',
        type=str,
        default='gpu',
        help='Use GPU or CPU')
    parser.add_argument(
        '--limit',
        type=int,
        default=20,
        help='Num of batc to HRank')
    args = parser.parse_args()

    print('==> Preparing data..')

    # 导入Cifar10数据库
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    trainset = torchvision.datasets.CIFAR10(root="./", train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root="./", train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
    # 用于裁剪的数据集
    pruningset = torchvision.datasets.CIFAR10(root="./", train=True, download=True, transform=transform_train)
    pruningloader = torch.utils.data.DataLoader(pruningset, batch_size=128, shuffle=True, num_workers=2)

    # 使用GPU还是CPU
    if args.gpu == 'gpu':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 创建神经网络模型
    print('==> Building Four model..')
    net1 = VGG(compress_rate=args.compress_rate)
    net1 = net1.to(device)
    net2 = VGG(compress_rate=args.compress_rate)
    net2 = net2.to(device)
    net3 = VGG(compress_rate=args.compress_rate)
    net3 = net3.to(device)
    net = Thinet_VGG()
    net = net.to(device)

    # 导入Google训练好的模型
    print('==> Loading weight..')
    # for var_name in net.state_dict():
    #     print(var_name,'\t',net.state_dict()[var_name].size())
    # checkpoint = torch.load('.\\vgg_16_bn.pt')
    if args.gpu == 'gpu':
        checkpoint = torch.load(r'./model/vgg_16_bn_cifar10.pt')
    else:
        checkpoint = torch.load(r'./model/vgg_16_bn_cifar10.pt', map_location=torch.device('cpu'))
    new_state_dict = OrderedDict()
    for k, v in checkpoint['state_dict'].items():
        new_state_dict[k.replace('module.', '')] = v
    net1.load_state_dict(new_state_dict)
    net2.load_state_dict(new_state_dict)
    net3.load_state_dict(new_state_dict)
    net.load_state_dict(new_state_dict)

    # 裁剪后的模型
    prun_net = Mask_VGG(net1, net1.compress_rate, device)
    random_net = Random_VGG(net2, net2.compress_rate, device)
    anti_net = Anti_VGG(net3, net3.compress_rate, device)
    # 优化器
    prun_optimizer = optim.SGD(net1.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    random_optimizer = optim.SGD(net2.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    anti_optimizer = optim.SGD(net3.parameters(), lr=0.0001, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.SGD(net.parameters(), lr=0.0001)
    # 调度机
    prun_scheduler = torch.optim.lr_scheduler.StepLR(prun_optimizer, step_size=10, gamma=0.5)
    random_scheduler = torch.optim.lr_scheduler.StepLR(random_optimizer, step_size=10, gamma=0.5)
    anti_scheduler = torch.optim.lr_scheduler.StepLR(anti_optimizer, step_size=10, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # 裁剪的初始化
    net.init_Thi()

    # 进行裁剪并微调
    print('==> Thinet Pruning..')
    inputs, _ = next(iter(pruningloader))
    net.compress(inputs, args.compress_rate, trainloader, criterion, optimizer, device)

    # 裁剪后的正确率
    print('==> After Thinet pruning..')
    test(net, testloader, device)
    # flops, params = profile(net, (inputs,))
    flops, params = get_flops.measure_model(net, device, 3, 32, 32, 1 - args.compress_rate)
    print('Flops:', flops)
    print('Params', params)


