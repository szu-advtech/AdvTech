import torchvision.models

import model_genotype as genotype
import models

import torch
import torch.utils.data
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
import sys
import numpy as np
import logging
import time
from tqdm import tqdm
from thop import profile


def evaluate(genome, init_channels=24, layers=8, epochs=25, device='cpu'):
    train_loader, test_loader = load_cifar10()
    if torch.cuda.is_available():
        dev = device
    else:
        dev = 'cpu'

    geno = genotype.decode(genome)
    net = models.NetworkCIFAR(C=init_channels,
                              num_classes=10,
                              layers=layers,
                              auxiliary=False,
                              genotype=geno,
                              increment=4)
    net = net.to(dev)
    parameters = filter(lambda p: p.requires_grad, net.parameters())
    criterion = nn.CrossEntropyLoss().to(dev)
    optimizer = optim.SGD(parameters, lr=0.025, momentum=0.9, weight_decay=3e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0)
    n_flops, n_params = 0, 0
    n_flops, n_params = get_flops(net, dev)
    # training
    for i in tqdm(range(epochs)):
        loss = _train(train_loader, net, optimizer, criterion, dev)
        scheduler.step()

    accuracy = _infer(test_loader, net, dev)
    return {
        'acc': accuracy,
        'flops': n_flops,
        'params': n_params
    }


def load_cifar10():

    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_transforms.transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),

    ])

    train_dataset = torchvision.datasets.CIFAR10(root='../data/', train=True, transform=train_transforms,
                                                 download=True)
    test_dataset = torchvision.datasets.CIFAR10(root='../data/', train=False, transform=test_transforms,
                                                download=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=2)
    return train_loader, test_loader


def get_flops(net, device):
    dummy_input = torch.rand([1, 3, 32, 32]).to(device)
    flops, params = profile(net, inputs=(dummy_input,), verbose=False)
    return flops, params


def _train(train_loader, net, optimizer, criterion, device):

    net.train()
    loss = 0
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        pred, _ = net(inputs)
        l = criterion(pred, targets)
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        l.backward()
        optimizer.step()
        loss += l.item()

    return loss


def _infer(test_loader, net, device):

    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs, _ = net(inputs)

            predict = outputs.argmax(dim=1)
            correct += predict.eq(targets).sum().item()
            total += targets.size(0)

    accuracy = 100. * correct / total
    return accuracy


# if __name__ == '__main__':
#     geno = genotype.test_genome
#     performance = evaluate(geno, epochs=1, device='cuda:0')
#     print(performance)
#
