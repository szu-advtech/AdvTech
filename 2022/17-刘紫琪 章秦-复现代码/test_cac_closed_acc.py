import argparse
import torch
import torchvision

import models
import torch.nn as nn
from torchvision import transforms, datasets
from dataload.tinyImagenet import TinyImageNet
import json

from efficientnet_pytorch import EfficientNet
from models.cac import CACModel
from utils import dataset_split, CACLoss
import torch.optim as optim
import time
import torch.nn.functional as F
import PIL

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def test_closedset_accuracy(model, test_known_loader, opt):
    start = time.time()
    # val
    cnt = 0
    epoch_loss = 0.
    epoch_acc = 0.

    model.eval()
    for step, (inputs, labels) in enumerate(test_known_loader):
        inputs = inputs.to(device)
        labels = torch.Tensor([opt.mapping[x] for x in labels]).long().to(device)

        outputs = model.forward1(inputs)
        cacLoss, anchorLoss, tupletLoss = CACLoss(outputs, labels, opt, device)

        # statistics
        _, predicted = outputs.min(1)
        epoch_loss = (cacLoss.item() * inputs.size(0) + cnt * epoch_loss) / (cnt + inputs.size(0))
        epoch_acc = (torch.sum(predicted == labels.data) + epoch_acc * cnt).double() / (cnt + inputs.size(0))

        cnt += inputs.size(0)

    print('test Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
    print('this epoch takes {} seconds.'.format(time.time() - start))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', choices=['CIFAR10', 'SVHN', 'CIFAR100', 'TinyImageNet'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_epoch', type=int, default=100)
    parser.add_argument('--split', type=int, default=0, choices=[0, 1, 2, 3, 4])

    parser.add_argument('--learning_rate', type=float, default=0.01, help='hyper-parameter: learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='hyper-parameter: weight decay')
    parser.add_argument('--_lambda', type=float, default=0.1, help='hyper-parameter: anchor loss weight lambda')
    parser.add_argument('--alpha', type=int, default=10, help='hyper-parameter: logit anchor magnitude')
    opt = parser.parse_args()

    print(opt)

    assert torch.cuda.is_available()

    # load dataset
    if opt.dataset == 'CIFAR10':
        trainval_idxs = "dataload/CIFAR10/trainval_idxs.json"
        class_splits = "dataload/CIFAR10/class_splits/" + str(opt.split) + ".json"

        with open(class_splits) as f:
            classSplits = json.load(f)
            known_classes = classSplits['Known']

        test_dataset = datasets.CIFAR10(root='data/cifar10', train=False, download=False,
                                         transform=transforms.Compose([
                                             transforms.Resize(32, PIL.Image.BICUBIC),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                                         ]))


        test_known_dataset = dataset_split(test_dataset, known_classes)

        print("size of training test known dataset cifar10", len(test_known_dataset))

        test_known_loader = torch.utils.data.DataLoader(test_known_dataset, batch_size=opt.batch_size, shuffle=True)

        opt.num_classes = 6
        mapping = [0 for i in range(10)]
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping

        save_path = "save_models/20221204/cifar10_cac.pth"

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

        test_dataset = datasets.SVHN(root='data/svhn', split='test', download=False,
                                    transform=transforms.Compose([
                                        transforms.Resize(32, PIL.Image.BICUBIC),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.4479, 0.4485, 0.4492), (0.2008, 0.1997, 0.1998)),
                                    ]))

        test_known_dataset = dataset_split(test_dataset, known_classes)

        print("size of training test known dataset svhn", len(test_known_dataset))

        test_known_loader = torch.utils.data.DataLoader(test_known_dataset, batch_size=opt.batch_size, shuffle=True)

        opt.num_classes = 6
        mapping = [0 for i in range(10)]
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping
        save_path = "save_models/tinyImageNet/SVHN19_cac.pth"

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

        test_dataset = torchvision.datasets.ImageFolder('data/tiny-imagenet-200/val',
                                                       transform=transforms.Compose([
                                                           transforms.Resize(64, PIL.Image.BICUBIC),
                                                           transforms.ToTensor(),
                                                           transforms.Normalize((0.4795, 0.4367, 0.3741), (0.2787, 0.2707, 0.2759)),
                                                       ]))

        test_known_dataset = dataset_split(test_dataset, known_classes)

        print("size of training test known dataset tiny-imagenet", len(test_known_dataset))

        test_known_loader = torch.utils.data.DataLoader(test_known_dataset, batch_size=opt.batch_size, shuffle=True)

        opt.num_classes = 20
        mapping = [0 for i in range(200)]
        known_classes.sort()
        for i, num in enumerate(known_classes):
            mapping[num] = i
        opt.mapping = mapping
        opt.lr_milestones = [150, 200]
        save_path = "save_models/tinyImageNet/TinyImageNet39_cac.pth"


    # load network efficientnet
    checkpoint = torch.load(save_path)

    # EfficientNet
    model_name = 'efficientnet-b5'
    embedding = EfficientNet.from_pretrained(model_name.lower())
    embedding._fc = nn.Sequential()
    classifier = nn.Linear(2048, opt.num_classes)
    model = CACModel(embedding, classifier, opt.num_classes)
    # print(model)

    model.load_state_dict(checkpoint)

    anchors = torch.diag(torch.Tensor([opt.alpha for i in range(opt.num_classes)]))
    model.set_anchors(anchors)

    model = model.to(device)

    test_closedset_accuracy(model, test_known_loader, opt)