import argparse
import torch
import torchvision.models

import models
import torch.nn as nn
from torchvision import transforms, datasets
from dataload.tinyImagenet import TinyImageNet
import json

from models.cac import CACModel
from utils import dataset_split, CACLoss
import torch.optim as optim
import time
import torch.nn.functional as F
import os
from torch.optim import lr_scheduler

from efficientnet_pytorch import EfficientNet
import PIL

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def train_cac(model, train_loader, val_loader, opt):
    # cifar10
    # optimizer = torch.optim.SGD(model.parameters(), opt.learning_rate, momentum=0.9, weight_decay=5e-6)
    # tiny-imagenet
    optimizer = torch.optim.SGD(model.parameters(), opt.learning_rate, momentum=0.9, weight_decay=10e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, float(opt.num_epoch))

    best_acc = 0.0

    for epoch in range(opt.num_epoch):
        start = time.time()
        print('Epoch {}/{}'.format(epoch, opt.num_epoch))
        print('-' * 10)

        # train
        cnt = 0
        epoch_loss = 0.
        epoch_acc = 0.

        model.train()
        for step, (inputs, labels) in enumerate(train_loader):
            with torch.set_grad_enabled(True):
                inputs = inputs.to(device)
                labels = torch.Tensor([opt.mapping[x] for x in labels]).long().to(device)

                optimizer.zero_grad()

                outputs = model.forward1(inputs)
                cacLoss, anchorLoss, tupletLoss = CACLoss(outputs, labels, opt, device)

                cacLoss.backward()
                optimizer.step()

                # statistics
                _, predicted = outputs.min(1)
                epoch_loss = (cacLoss.item() * inputs.size(0) + cnt * epoch_loss) / (cnt + inputs.size(0))
                epoch_acc = (torch.sum(predicted == labels.data) + epoch_acc * cnt).double() / (cnt + inputs.size(0))

                cnt += inputs.size(0)

        print('train Loss: {:.4f} anchorLoss: {:.4f} tupletLoss: {:.4f} Acc: {:.4f}'.format(epoch_loss, anchorLoss.item(), tupletLoss.item(), epoch_acc))
        scheduler.step()

        # val
        cnt = 0
        epoch_loss = 0.
        epoch_acc = 0.

        model.eval()
        for step, (inputs, labels) in enumerate(val_loader):
            inputs = inputs.to(device)
            labels = torch.Tensor([opt.mapping[x] for x in labels]).long().to(device)

            outputs = model.forward1(inputs)
            cacLoss, anchorLoss, tupletLoss = CACLoss(outputs, labels, opt, device)

            # statistics
            _, predicted = outputs.min(1)
            epoch_loss = (cacLoss.item() * inputs.size(0) + cnt * epoch_loss) / (cnt + inputs.size(0))
            epoch_acc = (torch.sum(predicted == labels.data) + epoch_acc * cnt).double() / (cnt + inputs.size(0))

            cnt += inputs.size(0)

        print('val Loss: {:.4f} anchorLoss: {:.4f} tupletLoss: {:.4f} Acc: {:.4f}'.format(epoch_loss, anchorLoss.item(),
                                                                                          tupletLoss.item(), epoch_acc))
        print('this epoch takes {} seconds.'.format(time.time() - start))

        saved_model_path = os.path.join(opt.output_folder, (opt.dataset+str(epoch)+'_cac.pth'))
        torch.save(model.state_dict(), saved_model_path)
        best_acc = epoch_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'SVHN', 'TinyImageNet', 'MNIST'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--split', type=int, default=0, choices=[0, 1, 2, 3, 4])

    parser.add_argument('--learning_rate', type=float, default=0.01, help='hyper-parameter: learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='hyper-parameter: weight decay')
    parser.add_argument('--_lambda', type=float, default=0.1, help='hyper-parameter: anchor loss weight lambda')
    parser.add_argument('--alpha', type=int, default=10, help='hyper-parameter: logit anchor magnitude')

    parser.add_argument('--output_folder', type=str, default='save_models/')
    opt = parser.parse_args()

    print(opt)

    assert torch.cuda.is_available()

    os.makedirs(opt.output_folder, exist_ok=True)

    # load dataset
    if opt.dataset == 'CIFAR10':
        trainval_idxs = "dataload/CIFAR10/trainval_idxs.json"
        class_splits = "dataload/CIFAR10/class_splits/"+str(opt.split)+".json"
        with open(trainval_idxs) as f:
            trainvalIdxs = json.load(f)
            train_idx = trainvalIdxs['Train']
            val_idx = trainvalIdxs['Val']

        with open(class_splits) as f:
            classSplits = json.load(f)
            known_classes = classSplits['Known']

        train_dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=False,
                                         transform=transforms.Compose([
                                             transforms.Resize(32, PIL.Image.BICUBIC),
                                             transforms.CenterCrop(32),
                                             transforms.RandomHorizontalFlip(0.5),
                                             transforms.RandomRotation(10),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4858, 0.4771, 0.4326), (0.2422, 0.2374, 0.2547)),
                                         ]))
        val_dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=False,
                                         transform=transforms.Compose([
                                             transforms.Resize(32, PIL.Image.BICUBIC),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4858, 0.4771, 0.4326), (0.2422, 0.2374, 0.2547)),
                                         ]))

        train_known_dataset = dataset_split(train_dataset, known_classes, train_idx)
        train_val_dataset = dataset_split(val_dataset, known_classes, val_idx)

        print("size of training known dataset cifar10", len(train_known_dataset))
        print("size of training val dataset cifar10", len(train_val_dataset))

        train_loader = torch.utils.data.DataLoader(train_known_dataset, batch_size=opt.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=opt.batch_size, shuffle=True)

        opt.num_classes = 6
        mapping = [0 for i in range(10)]
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping
        opt.lr_milestones = [150, 200]

    elif opt.dataset == 'SVHN':
        trainval_idxs = "dataload/SVHN/trainval_idxs.json"
        class_splits = "dataload/SVHN/class_splits/" + str(opt.split) + ".json"
        with open(trainval_idxs) as f:
            trainvalIdxs = json.load(f)
            train_idx = trainvalIdxs['Train']
            val_idx = trainvalIdxs['Val']

        with open(class_splits) as f:
            classSplits = json.load(f)
            known_classes = classSplits['Known']

        train_dataset = datasets.SVHN(root='data/svhn', download=False,
                                         transform=transforms.Compose([
                                             transforms.Resize(32, PIL.Image.BICUBIC),
                                             transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
                                             transforms.RandomHorizontalFlip(0.5),
                                             transforms.RandomRotation(10),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4479, 0.4485, 0.4492), (0.2008, 0.1997, 0.1998)),
                                         ]))
        val_dataset = datasets.SVHN(root='data/svhn', download=False,
                                       transform=transforms.Compose([
                                           transforms.Resize(32, PIL.Image.BICUBIC),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.4479, 0.4485, 0.4492), (0.2008, 0.1997, 0.1998)),
                                       ]))

        train_known_dataset = dataset_split(train_dataset, known_classes, train_idx)
        train_val_dataset = dataset_split(val_dataset, known_classes, val_idx)

        print("size of training known dataset svhn", len(train_known_dataset))
        print("size of training val dataset svhn", len(train_val_dataset))

        train_loader = torch.utils.data.DataLoader(train_known_dataset, batch_size=opt.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=opt.batch_size, shuffle=True)

        opt.num_classes = 6
        mapping = [0 for i in range(10)]
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping
        opt.lr_milestones = [150, 200]

    elif opt.dataset == 'MNIST':
        trainval_idxs = "dataload/MNIST/trainval_idxs.json"
        class_splits = "dataload/MNIST/class_splits/"+str(opt.split)+".json"
        with open(trainval_idxs) as f:
            trainvalIdxs = json.load(f)
            train_idx = trainvalIdxs['Train']
            val_idx = trainvalIdxs['Val']

        with open(class_splits) as f:
            classSplits = json.load(f)
            known_classes = classSplits['Known']

        train_dataset = datasets.MNIST(root='data/mnist', train=True, download=False,
                                         transform=transforms.Compose([
                                             transforms.Resize(32, PIL.Image.BICUBIC),
                                             transforms.RandomResizedCrop(32, scale=(0.7, 1.0)),
                                             transforms.RandomHorizontalFlip(0.5),
                                             transforms.RandomRotation(10),
                                             transforms.ToTensor(),
                                             transforms.Normalize(0.1321, 0.3101),
                                         ]))
        val_dataset = datasets.MNIST(root='data/mnist', train=True, download=False,
                                         transform=transforms.Compose([
                                             transforms.Resize(32, PIL.Image.BICUBIC),
                                             transforms.ToTensor(),
                                             transforms.Normalize(0.1321, 0.3101),
                                         ]))

        train_known_dataset = dataset_split(train_dataset, known_classes, train_idx)
        train_val_dataset = dataset_split(val_dataset, known_classes, val_idx)

        print("size of training known dataset mnist", len(train_known_dataset))
        print("size of training val dataset mnist", len(train_val_dataset))

        train_loader = torch.utils.data.DataLoader(train_known_dataset, batch_size=opt.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=opt.batch_size, shuffle=True)

        opt.num_classes = 6
        mapping = [0 for i in range(10)]
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping
        opt.lr_milestones = [150, 200]

    elif opt.dataset == 'TinyImageNet':
        trainval_idxs = "dataload/TinyImageNet/trainval_idxs.json"
        class_splits = "dataload/TinyImageNet/class_splits/" + str(opt.split) + ".json"
        with open(trainval_idxs) as f:
            trainvalIdxs = json.load(f)
            train_idx = trainvalIdxs['Train']
            val_idx = trainvalIdxs['Val']

        with open(class_splits) as f:
            classSplits = json.load(f)
            known_classes = classSplits['Known']

        train_dataset = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train',
                                                         transform=transforms.Compose([
                                             transforms.Resize(64, PIL.Image.BICUBIC),
                                             transforms.CenterCrop(64),
                                             transforms.RandomHorizontalFlip(0.5),
                                             transforms.RandomRotation(20),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4795, 0.4367, 0.3741), (0.2787, 0.2707, 0.2759)),
                                         ]))

        val_dataset = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train',
                                                       transform=transforms.Compose([
                                           transforms.Resize(64, PIL.Image.BICUBIC),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.4795, 0.4367, 0.3741), (0.2787, 0.2707, 0.2759)),
                                       ]))

        train_known_dataset = dataset_split(train_dataset, known_classes, train_idx)
        train_val_dataset = dataset_split(val_dataset, known_classes, val_idx)

        print("size of training known dataset tiny-imagenet", len(train_known_dataset))
        print("size of training val dataset tiny-imagenet", len(train_val_dataset))

        train_loader = torch.utils.data.DataLoader(train_known_dataset, batch_size=opt.batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(train_val_dataset, batch_size=opt.batch_size, shuffle=True)

        opt.num_classes = 20
        mapping = [0 for i in range(200)]
        known_classes.sort()
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping
        opt.lr_milestones = [150, 200]

    if opt.dataset == 'MNIST':
        # EfficientNet
        model_name = 'efficientnet-b2'
        embedding = EfficientNet.from_pretrained(model_name.lower(), in_channels=1)
        embedding._fc = nn.Sequential()
        classifier = nn.Linear(1408, opt.num_classes)
        model = CACModel(embedding, classifier, opt.num_classes)
    else:
        # EfficientNet
        model_name = 'efficientnet-b5'
        embedding = EfficientNet.from_pretrained(model_name.lower())
        embedding._fc = nn.Sequential()
        classifier = nn.Linear(2048, opt.num_classes)
        model = CACModel(embedding, classifier, opt.num_classes)

    anchors = torch.diag(torch.Tensor([opt.alpha for i in range(opt.num_classes)]))
    model.set_anchors(anchors)

    model = model.to(device)

    train_cac(model, train_loader, val_loader, opt)