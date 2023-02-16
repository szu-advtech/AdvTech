import argparse
import shutil

import torchvision.models

import train_operation as operation
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
import os

# =================================================================================
# set arguments for Terminal input
parser = argparse.ArgumentParser("Train an architecture created by genome")
parser.add_argument('--genofile', type=str, default=None, help='genome file path')
parser.add_argument('--n_geno', type=int, default=1, help='No. of genome in file')

parser.add_argument('--save', type=str, default='train', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')

parser.add_argument('--n_blocks', type=int, default=5, help='number of blocks in a cell')
parser.add_argument('--n_nodes', type=int, default=4, help='number of nodes per phases')

parser.add_argument('--init_channels', type=int, default=36, help='channels of filters for first cell')
parser.add_argument('--layers', type=int, default=20, help='number of layers of the networks')
parser.add_argument('--epochs', type=int, default=600, help='training epochs for each individual')

parser.add_argument('--device', type=str, default='cuda:0', help='GPU/CPU device selected')

args = parser.parse_args()
args.save = '{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))


device = args.device if torch.cuda.is_available() else 'cpu'

sys.path.insert(0, '/data/sunliang/sunliang/projects/nas')

# =================================================================================
def mkdir_save(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    # print(f'Experiment directory: {path}')

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


mkdir_save(args.save)
log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def cifar10_transforms():
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_transform.transforms.append(transforms.Normalize(CIFAR_MEAN, CIFAR_STD))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
    ])
    return train_transform, valid_transform


def load_genome(file, number=1):
    genome = []
    s = ""
    with open(file) as f:
        for i in range(number):
            s = f.readline()
    genome = list(map(int, s.split(' ')[:-1]))
    return genome


def main():

    writer = SummaryWriter(logdir=os.path.join(args.save, 'logs'))

    seed = args.seed
    torch.cuda.set_device(device)
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # genome = genotype.test_genome
    # geno = genotype.decode(genome)
    # geno = genotype.TestNet
    genome = load_genome(args.genofile, args.n_geno)
    geno = genotype.decode(genome)
    net = models.NetworkCIFAR(C=args.init_channels, num_classes=10, layers=args.layers, auxiliary=True, genotype=geno).to(device)

    train_transform, valid_transform = cifar10_transforms()
    train_set = torchvision.datasets.CIFAR10(root='../data/', train=True, transform=train_transform, download=True)
    valid_set = torchvision.datasets.CIFAR10(root='../data/', train=False, transform=valid_transform, download=True)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=96, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=96, shuffle=False, num_workers=2, pin_memory=True)

    dummy_input = torch.rand(2, 3, 32, 32).to(device)
    writer.add_graph(net, [dummy_input, ])
    n_flops, n_params = operation.get_flops(net, device)

    logging.info(f'genome: {genome}\n\n')
    logging.info(f'genotype: {geno}\n\n')
    logging.info(f'FLOPs: {n_flops / 1e6} M\n')
    logging.info(f'Params: {n_params / 1e6} M\n')

    with open(os.path.join(args.save, 'result'), 'a') as f:
        f.write(f'genome: {genome}\n\n')
        f.write(f'genotype: {geno}\n\n')
        f.write(f'FLOPs: {n_flops / 1e6} M\n')
        f.write(f'Params: {n_params / 1e6} M\n')

    epochs = args.epochs

    parameters = filter(lambda p: p.requires_grad, net.parameters())
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(parameters, lr=0.025, momentum=0.9, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, eta_min=0.0)
    # optimizer = optim.Adam(parameters, lr=0.03)
    valid_acc = 0.0
    for epoch in range(epochs):
        logging.info(f'epoch {epoch} lr {scheduler.get_lr()[0]}')

        train_loss, train_acc = train(train_loader, net, criterion, optimizer)
        valid_loss, valid_acc = infer(valid_loader, net, criterion)
        scheduler.step()
        torch.save(net.state_dict(), os.path.join(args.save, 'weight.pth'))

        logging.info(f'epoch {epoch+1}/{epochs}\tloss:{train_loss}\tacc:{valid_acc}')
        writer.add_scalar('loss', train_loss, epoch+1)
        writer.add_scalar('Accuracy', valid_acc, epoch+1)

    writer.close()
    with open(os.path.join(args.save, 'result'), 'a') as f:
        f.write(f'Accuracy: {valid_acc}\n')
    logging.info(f"output result file: {os.path.join(args.save, 'result')}")
    logging.info(f"save model weight in file: {os.path.join(args.save, os.path.join(args.save, 'weight.pth'))}")


def train(train_loader, net, criterion, optimizer, auxiliary=False):
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    for step, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs, outputs_aux = net(inputs)
        loss = criterion(outputs, targets)

        if auxiliary:
            loss_aux = criterion(outputs_aux, targets)
            loss += loss_aux * 0.4

        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 5)
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    logging.info('train acc %f', 100. * correct / total)

    return train_loss / total, 100.*correct/total


def infer(valid_loader, net, criterion):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    acc = 100. * correct / total
    logging.info('valid acc %f', 100. * correct / total)

    return test_loss / total, acc


if __name__ == '__main__':
    main()

