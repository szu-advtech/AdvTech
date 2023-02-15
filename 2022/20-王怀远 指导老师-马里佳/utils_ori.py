#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import torchxrayvision as xrv
import copy
import torch
from torchvision import datasets, transforms
from sampling_ori import mnist_iid, mnist_noniid, mnist_noniid_unequal
from sampling_ori import cifar_iid, cifar_noniid, cifar_noniid_new, kvasir_noniid_new
from sampling_ori import covid_noniid
import pdb

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
from torch.autograd import Variable
from options import args_parser

args = args_parser()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


class DriveData(Dataset):
    __xs = []
    __ys = []

    def __init__(self, folder_dataset, transform=None):
        self.transform = transform
        # Open and load text file including the whole training data
        with open(folder_dataset) as f:
            for line in f:
                # Image path
                self.__xs.append(line.split()[0])
                # Steering wheel label
                self.__ys.append(np.int64(line.split()[1]))
        self.targets = torch.from_numpy(np.asarray(self.__ys)).view(-1)

    # Override to give PyTorch access to any image on the dataset

    def __getitem__(self, index):
        img = Image.open(self.__xs[index])
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        # Convert image and label to torch tensors
        img = Variable(img)
        label = torch.from_numpy(np.asarray(self.__ys[index]).reshape([1, 1]))
        return img, label

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)


COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}


def read_filepaths(file):
    paths, labels = [], []
    with open(file, 'r') as f:
        lines = f.read().splitlines()

        for idx, line in enumerate(lines):
            if ('/ c o' in line):
                break
            subjid, path, label, class_image = line.split(' ')

            paths.append(path)
            labels.append(label)
    return paths, labels


'''
class COVIDxDataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, mode, train_transformer, val_transformer, n_classes=3, dataset_path='datax', dim=(224, 224)):
        self.root = str(dataset_path) + '/' + mode + '/'

        self.CLASSES = n_classes
        self.dim = dim
        self.COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
        self.targets = []
        testfile = 'test_split.txt'
        trainfile = 'train_split.txt'
        if (mode == 'train'):
            self.paths, self.labels = read_filepaths(trainfile)
            self.transform = train_transformer
        elif (mode == 'test'):
            self.paths, self.labels = read_filepaths(testfile)
            self.transform = val_transformer
        print("{} examples =  {}".format(mode, len(self.paths)))
        self.mode = mode
        #label_tensor_1 = torch.tensor(self.COVIDxDICT[self.labels], dtype=torch.long)
        self.targets=[self.COVIDxDICT[i] for i in self.labels]
        #print(self.targets)
        #pdb.set_trace()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image_tensor = self.load_image(self.root + self.paths[index], self.dim, augmentation=self.mode)
        label_tensor = torch.tensor(self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path, dim, augmentation='test'):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)

        image_tensor = self.transform(image)

        return image_tensor
'''
normalize_covid = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform_covid = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop((224), scale=(0.5, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_covid
])
val_transform_covid = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize_covid
])


class COVIDxDataset(Dataset):
    """
    Code for reading the COVIDxDataset
    """

    def __init__(self, mode, n_classes=3, dataset_path='./datax', dim=(224, 224)):
        self.root = str(dataset_path) + '/' + mode + '/'

        self.CLASSES = n_classes
        self.dim = dim
        self.targets = []
        self.COVIDxDICT = {'pneumonia': 0, 'normal': 1, 'COVID-19': 2}
        testfile = './datax/test_COVIDx9A.txt'
        trainfile = './datax/train_COVIDx9A.txt'
        if mode == 'train':
            self.paths, self.labels = read_filepaths(trainfile)
            self.transform = train_transform_covid
        elif mode == 'test':
            self.paths, self.labels = read_filepaths(testfile)
            self.transform = val_transform_covid
        print("{} examples =  {}".format(mode, len(self.paths)))
        # print(self.labels)
        print(self.COVIDxDICT)
        self.targets = [self.COVIDxDICT[i] for i in self.labels]
        self.mode = mode

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):

        image_tensor = self.load_image(
            self.root + self.paths[index], self.dim, augmentation=self.mode)
        label_tensor = torch.tensor(
            self.COVIDxDICT[self.labels[index]], dtype=torch.long)

        return image_tensor, label_tensor

    def load_image(self, img_path, dim, augmentation='test'):
        if not os.path.exists(img_path):
            print("IMAGE DOES NOT EXIST {}".format(img_path))
        image = Image.open(img_path).convert('RGB')
        image = image.resize(dim)

        image_tensor = self.transform(image)

        return image_tensor


def get_dataset(args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    """
    # print(789)
    # print(args.dataset)

    if args.dataset == 'cifar':
        data_dir = '../data/cifar/'
        # apply_transform = transforms.Compose(
        #    [transforms.ToTensor(),
        #     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])

        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True,
                                         transform=transform_test)

        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True,
                                        transform=transform_test)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = cifar_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                raise NotImplementedError()
            else:
                # Chose euqal splits for every user
                #user_groups = cifar_noniid(train_dataset, args.num_users)
                #from collections import Counter
                #la = np.array(train_dataset.targets)
                user_groups = cifar_noniid_new(
                    train_dataset, args.num_users, args.Lam, args.num_chunk)

    elif args.dataset == "covid":

        train_dataset = COVIDxDataset(mode='train',  n_classes=3, dataset_path="./datax",
                                      dim=(224, 224))
        test_dataset = COVIDxDataset(mode='test', n_classes=3, dataset_path="./datax",
                                     dim=(224, 224))
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            user_groups = covid_noniid(
                train_dataset, args.num_users, args.Lam, args.num_chunk)

    elif args.dataset == 'kvasir':
        # print(1234)

        prep = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        train_dataset = DriveData("train.txt", prep)

        test_dataset = DriveData("val.txt", prep)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            user_groups = kvasir_noniid_new(
                train_dataset, args.num_users, args.Lam, args.num_chunk)

    elif args.dataset == 'mnist' or 'fmnist':

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        # print(456)

        if args.dataset == 'mnist':
            data_dir = '../data/mnist/'
            train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                           transform=apply_transform)

            test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                          transform=apply_transform)

        else:
            data_dir = '../data/fmnist/'
            train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True,
                                                  transform=apply_transform)

            test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True,
                                                 transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(
                    train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                #user_groups = mnist_noniid(train_dataset, args.num_users)
                user_groups = mnist_noniid(train_dataset, args.num_users)

    elif args.dataset == 'SST2':

        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))])

        train_dataset = datasets.MNIST(data_dir, train=True, download=True,
                                       transform=apply_transform)

        test_dataset = datasets.MNIST(data_dir, train=False, download=True,
                                      transform=apply_transform)

        # sample training data amongst users
        if args.iid:
            # Sample IID user data from Mnist
            user_groups = mnist_iid(train_dataset, args.num_users)
        else:
            # Sample Non-IID user data from Mnist
            if args.unequal:
                # Chose uneuqal splits for every user
                user_groups = mnist_noniid_unequal(
                    train_dataset, args.num_users)
            else:
                # Chose euqal splits for every user
                user_groups = mnist_noniid(train_dataset, args.num_users)

    return train_dataset, test_dataset, user_groups


def average_weights_online(w_avg, w_new, nth):
    """
    Returns the average of the weights online.
    follows the equation avg_t = avg_t_1 + (a_t - avg_t_1) / t
    """
    for key in w_avg.keys():
        w_avg[key] = w_avg[key] + (w_new[key] - w_avg[key]) / nth
    return w_avg


def weights2cpu(we):
    """
    Returns the cpu weights.
    """
    for key in we.keys():
        we[key] = we[key].cpu()
    return we


def weights2gpu(we):
    """ Returns the cpu weights.
    """
    for key in we.keys():
        we[key] = we[key].cuda()
    return we


def average_weights(w):
    """
    Returns the average of the weights.
    """
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[key] += w[i][key]
        w_avg[key] = torch.div(w_avg[key], len(w))
    return w_avg

# The following function is used in


def get_local_weights_names(dataset_here, epoch, scheduler, global_weights):
    if dataset_here == "mnist":
        if epoch <= scheduler[0]:
            local_weights_names = [s for s in global_weights.keys(
            ) if s.startswith('fc') or s.startswith('conv2')]
        elif epoch <= scheduler[1]:
            local_weights_names = [
                s for s in global_weights.keys() if s.startswith('fc')]
        else:
            local_weights_names = [
                s for s in global_weights.keys() if s.startswith('fc2')]

    if dataset_here == "fmnist":
        if epoch <= scheduler[0]:
            local_weights_names = [s for s in global_weights.keys(
            ) if s.startswith('fc') or s.startswith('layer2')]
        else:
            local_weights_names = [
                s for s in global_weights.keys() if s.startswith('fc')]

    return local_weights_names


def get_local_weights_names_new(dataset_here, epoch, scheduler, global_weights):
    if dataset_here == "mnist":
        if epoch % 3 == 0:
            local_weights_names = [s for s in global_weights.keys(
            ) if s.startswith('fc') or s.startswith('conv2')]
        elif epoch % 3 == 1:
            local_weights_names = [
                s for s in global_weights.keys() if s.startswith('fc')]
        else:
            local_weights_names = [
                s for s in global_weights.keys() if s.startswith('fc2')]

    if dataset_here == "fmnist":
        if epoch % 2 == 0:
            local_weights_names = [s for s in global_weights.keys(
            ) if s.startswith('fc') or s.startswith('layer2')]
        else:
            local_weights_names = [
                s for s in global_weights.keys() if s.startswith('fc')]

    return local_weights_names


def exp_details(args):
    print('\nExperimental details:')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    if args.iid:
        print('    IID')
    else:
        print('    Non-IID')
    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')
    return
