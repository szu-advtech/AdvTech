import copy
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, autograd


class Node:
    def __init__(self, device, epoch, dataset_train, dataset_test, lr):
        self.epoch = epoch
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.device = device
        self.lr = lr
        self.ldr_train = DataLoader(self.dataset_train, batch_size=epoch, shuffle=True)
        self.loss_func = nn.CrossEntropyLoss()

    def train(self, net_glob):
        optimizer = torch.optim.SGD(net_glob.parameters(), lr=self.lr)
        for iter in range(self.epoch):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                net_glob.train()
                images, labels = images.to(self.device), labels.to(self.device)
                net_glob.zero_grad()
                log_probs = net_glob(images)
                loss = self.loss_func(log_probs, labels)
                loss.backward()
                optimizer.step()
                if batch_idx % 10 == 0:
                    print('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        iter, batch_idx * len(images), len(self.ldr_train.dataset),
                              100. * batch_idx / len(self.ldr_train), loss.item()))
