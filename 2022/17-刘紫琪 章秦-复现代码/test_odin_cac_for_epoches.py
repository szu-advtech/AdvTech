# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import torchvision
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from torch.autograd import Variable
import argparse
import torch
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
from efficientnet_pytorch import EfficientNet
import PIL

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'MNIST', 'TinyImageNet', 'SVHN'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=50)
    parser.add_argument('--split', type=int, default=0, choices=[0, 1, 2, 3, 4])

    parser.add_argument('--learning_rate', type=float, default=0.01, help='hyper-parameter: learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='hyper-parameter: weight decay')
    parser.add_argument('--_lambda', type=float, default=0.1, help='hyper-parameter: anchor loss weight lambda')
    parser.add_argument('--alpha', type=int, default=10, help='hyper-parameter: logit anchor magnitude')

    parser.add_argument('--magnitude', default=0.0014, type=float, help='perturbation magnitude')
    parser.add_argument('--temperature', default=1000, type=int, help='temperature scaling')
    parser.add_argument('--output_folder', type=str, default='save_models/')
    opt = parser.parse_args()

    print(opt)

    assert torch.cuda.is_available()

    os.makedirs(opt.output_folder, exist_ok=True)

    # load dataset
    if opt.dataset == 'CIFAR10':
        trainval_idxs = "dataload/CIFAR10/trainval_idxs.json"
        class_splits = "dataload/CIFAR10/class_splits/" + str(opt.split) + ".json"
        with open(trainval_idxs) as f:
            trainvalIdxs = json.load(f)
            train_idx = trainvalIdxs['Train']
            val_idx = trainvalIdxs['Val']

        with open(class_splits) as f:
            classSplits = json.load(f)
            known_classes = classSplits['Known']
            unknown_classes = classSplits['Unknown']

        test_known_dataset = datasets.CIFAR10(root='data/cifar10', train=False, download=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(32, PIL.Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                        ]))
        test_unknown_dataset = datasets.CIFAR10(root='data/cifar10', train=False, download=False,
                                              transform=transforms.Compose([
                                                  transforms.Resize(32, PIL.Image.BICUBIC),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                       (0.2023, 0.1994, 0.2010)),
                                              ]))
        train_unknown_dataset = datasets.CIFAR10(root='data/cifar10', train=True, download=False,
                                                transform=transforms.Compose([
                                                    transforms.Resize(32, PIL.Image.BICUBIC),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                         (0.2023, 0.1994, 0.2010)),
                                                ]))

        test_known_dataset = dataset_split(test_known_dataset, known_classes)
        test_unknown_dataset = dataset_split(test_unknown_dataset, unknown_classes)
        train_unknown_dataset = dataset_split(train_unknown_dataset, unknown_classes)

        print("size of test known dataset cifar10", len(test_known_dataset))
        print("size of test unknown dataset cifar10", len(test_unknown_dataset))
        print("size of train unknown dataset cifar10", len(train_unknown_dataset))

        test_known_loader = torch.utils.data.DataLoader(test_known_dataset, batch_size=opt.batch_size, shuffle=True)
        test_unknown_loader = torch.utils.data.DataLoader(test_unknown_dataset, batch_size=opt.batch_size, shuffle=True)
        train_unknown_loader = torch.utils.data.DataLoader(train_unknown_dataset, batch_size=opt.batch_size, shuffle=True)

        opt.num_classes = 6
        mapping = [0 for i in range(10)]
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping
        opt.lr_milestones = [150, 200]

        save_path = "save_models/20221204/cifar10_cac.pth"

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
            unknown_classes = classSplits['Unknown']

        test_known_dataset = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val',
                                                       transform=transforms.Compose([
                                                           transforms.Resize(64, PIL.Image.BICUBIC),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.4795, 0.4367, 0.3741),
                                                                                (0.2787, 0.2707, 0.2759)),
                                                       ]))
        test_unknown_dataset = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val',
                                                              transform=transforms.Compose([
                                                                  transforms.Resize(64, PIL.Image.BICUBIC),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.4795, 0.4367, 0.3741),
                                                                                       (0.2787, 0.2707, 0.2759)),
                                                              ]))
        train_unknown_dataset = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/train',
                                                              transform=transforms.Compose([
                                                                  transforms.Resize(64, PIL.Image.BICUBIC),
                                                                  transforms.ToTensor(),
                                                                  transforms.Normalize((0.4795, 0.4367, 0.3741),
                                                                                       (0.2787, 0.2707, 0.2759)),
                                                              ]))

        test_known_dataset = dataset_split(test_known_dataset, known_classes)
        test_unknown_dataset = dataset_split(test_unknown_dataset, unknown_classes)
        train_unknown_dataset = dataset_split(train_unknown_dataset, unknown_classes)

        print("size of test known dataset tiny-imagenet", len(test_known_dataset))
        print("size of test unknown dataset tiny-imagenet", len(test_unknown_dataset))
        print("size of train unknown dataset tiny-imagenet", len(train_unknown_dataset))

        test_known_loader = torch.utils.data.DataLoader(test_known_dataset, batch_size=opt.batch_size, shuffle=True)
        test_unknown_loader = torch.utils.data.DataLoader(test_unknown_dataset, batch_size=opt.batch_size, shuffle=True)
        train_unknown_loader = torch.utils.data.DataLoader(train_unknown_dataset, batch_size=opt.batch_size,
                                                           shuffle=True)

        opt.num_classes = 20
        known_classes.sort()
        mapping = [0 for i in range(200)]
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping
        opt.lr_milestones = [150, 200]

        save_path = "save_models/20221204/tiny_imagenet_cac_0.4.pth"

    elif opt.dataset == 'MNIST':
        trainval_idxs = "dataload/MNIST/trainval_idxs.json"
        class_splits = "dataload/MNIST/class_splits/" + str(opt.split) + ".json"
        with open(trainval_idxs) as f:
            trainvalIdxs = json.load(f)
            train_idx = trainvalIdxs['Train']
            val_idx = trainvalIdxs['Val']

        with open(class_splits) as f:
            classSplits = json.load(f)
            known_classes = classSplits['Known']
            unknown_classes = classSplits['Unknown']

        test_known_dataset = datasets.MNIST(root='data/mnist', train=False, download=False,
                                        transform=transforms.Compose([
                                            transforms.Resize(32, PIL.Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize(0.1321, 0.3101),
                                        ]))
        test_unknown_dataset = datasets.MNIST(root='data/mnist', train=False, download=False,
                                              transform=transforms.Compose([
                                                  transforms.Resize(32, PIL.Image.BICUBIC),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(0.1321, 0.3101),
                                              ]))
        train_unknown_dataset = datasets.MNIST(root='data/mnist', train=True, download=False,
                                                transform=transforms.Compose([
                                                    transforms.Resize(32, PIL.Image.BICUBIC),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(0.1321, 0.3101),
                                                ]))

        test_known_dataset = dataset_split(test_known_dataset, known_classes)
        test_unknown_dataset = dataset_split(test_unknown_dataset, unknown_classes)
        train_unknown_dataset = dataset_split(train_unknown_dataset, unknown_classes)

        print("size of test known dataset mnist", len(test_known_dataset))
        print("size of test unknown dataset mnist", len(test_unknown_dataset))
        print("size of train unknown dataset mnist", len(train_unknown_dataset))

        test_known_loader = torch.utils.data.DataLoader(test_known_dataset, batch_size=opt.batch_size, shuffle=True)
        test_unknown_loader = torch.utils.data.DataLoader(test_unknown_dataset, batch_size=opt.batch_size, shuffle=True)
        train_unknown_loader = torch.utils.data.DataLoader(train_unknown_dataset, batch_size=opt.batch_size, shuffle=True)

        opt.num_classes = 6
        mapping = [0 for i in range(10)]
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping
        opt.lr_milestones = [150, 200]

        save_path = "save_models/20221204/MNIST_cac.pth"

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
            unknown_classes = classSplits['Unknown']

        test_known_dataset = datasets.SVHN(root='data/svhn', split = 'test', download=True,
                                    transform=transforms.Compose([
                                        transforms.Resize(32, PIL.Image.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4479, 0.4485, 0.4492), (0.2008, 0.1997, 0.1998)),
                                    ]))
        test_unknown_dataset = datasets.SVHN(root='data/svhn', split='test', download=True,
                                           transform=transforms.Compose([
                                               transforms.Resize(32, PIL.Image.BICUBIC),
                                               transforms.ToTensor(),
                                               transforms.Normalize((0.4479, 0.4485, 0.4492), (0.2008, 0.1997, 0.1998)),
                                           ]))
        train_unknown_dataset = datasets.SVHN(root='data/svhn', download=False,
                                    transform=transforms.Compose([
                                        transforms.Resize(32, PIL.Image.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4479, 0.4485, 0.4492), (0.2008, 0.1997, 0.1998)),
                                    ]))

        test_known_dataset = dataset_split(test_known_dataset, known_classes)
        test_unknown_dataset = dataset_split(test_unknown_dataset, unknown_classes)
        train_unknown_dataset = dataset_split(train_unknown_dataset, unknown_classes)

        print("size of test known dataset svhn", len(test_known_dataset))
        print("size of test unknown dataset svhn", len(test_unknown_dataset))
        print("size of train unknown dataset svhn", len(train_unknown_dataset))

        test_known_loader = torch.utils.data.DataLoader(test_known_dataset, batch_size=opt.batch_size, shuffle=True)
        test_unknown_loader = torch.utils.data.DataLoader(test_unknown_dataset, batch_size=opt.batch_size, shuffle=True)
        train_unknown_loader = torch.utils.data.DataLoader(train_unknown_dataset, batch_size=opt.batch_size,
                                                           shuffle=True)

        opt.num_classes = 6
        mapping = [0 for i in range(10)]
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping
        opt.lr_milestones = [150, 200]


    # validation softmax+threshold:
    print("validation softmax+threshold:")

    # validation - calculating scores

    to_np = lambda x: x.data.cpu().numpy()

    for epoch in range(50, 100):
        save_path = "save_models/mnist/MNIST"+str(epoch)+"_cac.pth"
        # save_path = "save_models/svhn/SVHN" + str(epoch) + "_cac.pth"
        # save_path = "save_models/tinyImageNet/TinyImageNet" + str(epoch) + "_cac.pth"
        # save_path = "save_models/tinyImageNet/SVHN" + str(epoch) + "_cac.pth"

        if opt.dataset == 'MNIST':
            # EfficientNet
            model_name = 'efficientnet-b5'
            embedding = EfficientNet.from_pretrained(model_name.lower(), in_channels=1)
            embedding._fc = nn.Sequential()
            classifier = nn.Linear(2048, opt.num_classes)
            model = CACModel(embedding, classifier, opt.num_classes)
        else:
            # EfficientNet
            model_name = 'efficientnet-b5'
            embedding = EfficientNet.from_pretrained(model_name.lower())
            embedding._fc = nn.Sequential()
            classifier = nn.Linear(2048, opt.num_classes)
            model = CACModel(embedding, classifier, opt.num_classes)

        checkpoint = torch.load(save_path)
        model.load_state_dict(checkpoint)

        anchors = torch.diag(torch.Tensor([opt.alpha for i in range(opt.num_classes)]))
        model.set_anchors(anchors)

        model = model.to(device)

        ND_labels = np.hstack(
            [np.zeros(len(test_known_dataset)), np.ones(len(test_unknown_dataset)),
             np.ones(len(train_unknown_dataset))])
        scores_base = np.hstack(
            [np.zeros(len(test_known_dataset)), np.zeros(len(test_unknown_dataset)),
             np.zeros(len(train_unknown_dataset))])
        scores = np.hstack(
            [np.zeros(len(test_known_dataset)), np.zeros(len(test_unknown_dataset)),
             np.zeros(len(train_unknown_dataset))])

        idx = 0
        for loader in [test_known_loader, test_unknown_loader, train_unknown_loader]:
            for step, (inputs, labels) in enumerate(loader):
                inputs = Variable(inputs.to(device), requires_grad=True)
                labels = torch.Tensor([opt.mapping[x] for x in labels]).long().to(device)

                # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
                outputs = model.forward1(inputs)
                # Calculating the confidence of the output, no perturbation added here, no temperature scaling used
                nnOutputs = to_np(outputs)
                scores_base[idx:idx + len(nnOutputs)] = np.min(nnOutputs, axis=1)

                # Using temperature scaling
                outputs = outputs / opt.temperature

                # Calculating the perturbation we need to add, that is,
                # the sign of gradient of cross entropy loss w.r.t. input
                cacLoss, anchorLoss, tupletLoss = CACLoss(outputs, labels, opt, device)
                cacLoss.backward()

                # Normalizing the gradient to binary in {0, 1}
                gradient = torch.ge(inputs.grad.data, 0)
                gradient = (gradient.float() - 0.5) * 2
                # Normalizing the gradient to the same space of image
                if opt.dataset == 'MNIST':
                    gradient = (gradient) / (66.7 / 255.0)
                else:
                    gradient[:][0] = (gradient[:][0]) / (63.0 / 255.0)
                    gradient[:][1] = (gradient[:][1]) / (62.1 / 255.0)
                    gradient[:][2] = (gradient[:][2]) / (66.7 / 255.0)
                # Adding small perturbations to images
                tempInputs = torch.add(inputs.data, -opt.magnitude, gradient)

                outputs = model.forward1(Variable(tempInputs))
                outputs = outputs / opt.temperature
                # Calculating the confidence after adding perturbations
                nnOutputs = outputs.data.cpu()
                nnOutputs = nnOutputs.numpy()
                # nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
                # nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

                scores[idx:idx + len(nnOutputs)] = np.min(nnOutputs, axis=1)
                idx += len(nnOutputs)

        print("epoch: ", epoch)
        print('BASELINE: AUC ROC: %f' % roc_auc_score(ND_labels, scores_base))
        print('ODIN: AUC ROC: %f' % roc_auc_score(ND_labels, scores))
        print("================================================================")