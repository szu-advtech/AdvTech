#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pdb
from torch.optim.lr_scheduler import ReduceLROnPlateau
from options import args_parser
from collections import Counter
import random
from sklearn.metrics import confusion_matrix
args = args_parser()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class LocalUpdate(object):
    def __init__(self, args, dataset, idxs, logger):
        self.args = args
        self.logger = logger
        self.trainloader, self.testloader = self.train_val_test(
            dataset, list(idxs)
        )
        self.device = 'cuda' if args.gpu else 'cpu'

        self.criterion = nn.NLLLoss().to(self.device)
        if args.dataset == "covid":
            self.criterion = nn.CrossEntropyLoss(reduction='mean')

    def train_val_test(self, dataset, idxs):
        """
        Returns train, validation and test dataloaders for a given dataset
        and user indexes.
        """
        # split indexes for train, validation, and test (80, 10, 10)

        random.shuffle(idxs)
        idxs_train = idxs[:int(0.8*len(idxs))]
        idxs_val = idxs[int(0.8*len(idxs)):int(0.9*len(idxs))]
        idxs_test = idxs[int(0.9*len(idxs)):]

        idxs_train_covid = idxs[:int(0.7 * len(idxs))]
        idxs_test_covid = idxs[int(0.7 * len(idxs)):]

        trainloader = DataLoader(DatasetSplit(dataset, idxs_train),
                                 batch_size=self.args.local_bs, shuffle=False)

        if self.args.dataset == "kvasir":
            validloader = DataLoader(DatasetSplit(dataset, idxs_val),
                                     batch_size=int(len(idxs_val) / 1), shuffle=False)
            testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                    batch_size=int(len(idxs_test) / 1), shuffle=False)

        elif self.args.dataset == "covid":
            testloader = DataLoader(DatasetSplit(dataset, idxs_test_covid),
                                    batch_size=self.args.local_bs, shuffle=False)

        else:
            testloader = DataLoader(DatasetSplit(dataset, idxs_test),
                                    batch_size=int(len(idxs_test) / 10), shuffle=False)
        return trainloader, testloader

    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = [1]

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=self.args.weight_decay)

        scheduler = ReduceLROnPlateau(
            optimizer, factor=0.5, patience=2, min_lr=1e-6, verbose=True)
        for iter in range(self.args.local_ep):
            batch_loss = []

            for batch_idx, (images, labels) in enumerate(self.trainloader):
                # pdb.set_trace()
                # print(labels.size())
                labels = labels.view(-1)
                images, labels = images.to(self.device), labels.to(self.device)

                # print(images.size())

                model.zero_grad()
                log_probs = model(images)

                loss = self.criterion(log_probs, labels)

                loss.backward()
                optimizer.step()

                if self.args.verbose and (batch_idx % 10 == 0):
                    print('| Global Round : {} | Local Epoch : {} | [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        global_round, iter, batch_idx * len(images),
                        len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))
                self.logger.add_scalar('loss', loss.item())
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            scheduler.step(sum(batch_loss)/len(batch_loss))

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss)

    def inference(self, model, idx_epoch):
        """ Returns the inference accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0

        total_pred = list()
        total_lab = list()
        test_la = []
        pred_labels_la = []
        iii = 0
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                labels = labels.view(-1)

                # Inference
                outputs = model(images)
                batch_loss = self.criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total_pred.extend(pred_labels.cpu().numpy().tolist())
                total_lab.extend(labels.cpu().numpy().tolist())
                total += len(labels)
                test_la = test_la + labels.cpu().numpy().tolist()
                pred_labels_la = pred_labels_la + pred_labels.cpu().numpy().tolist()
                iii = iii + 1

        accuracy = correct / total
        confusion_MA = confusion_matrix(test_la, pred_labels_la)

        return accuracy, loss, confusion_MA

    def test_inference(args, model, test_dataset, idx_epoch):
        """ Returns the test accuracy and loss.
        """

        model.eval()
        loss, total, correct = 0.0, 0.0, 0.0
        test_la = []
        pred_labels_la = []
        iii = 0

        if args.gpu == '0':
            device = 'cuda:0'
        if args.gpu == '1':
            device = 'cuda:1'
        criterion = nn.NLLLoss().to(device)
        if args.dataset == "covid" or "kvasir":
            criterion = nn.CrossEntropyLoss(reduction='mean')
        testloader = DataLoader(test_dataset, batch_size=5,
                                shuffle=False)
        if args.dataset == "kvasir":
            testloader = DataLoader(test_dataset, batch_size=50,
                                    shuffle=False)

        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(testloader):
                images, labels = images.to(device), labels.to(device)
                labels = labels.view(-1)

                # Inference
                outputs = model(images)
                batch_loss = criterion(outputs, labels)
                loss += batch_loss.item()

                # Prediction
                _, pred_labels = torch.max(outputs, 1)
                pred_labels = pred_labels.view(-1)
                correct += torch.sum(torch.eq(pred_labels, labels)).item()
                total += len(labels)
                test_la = test_la + labels.cpu().numpy().tolist()
                pred_labels_la = pred_labels_la + pred_labels.cpu().numpy().tolist()
                iii = iii + 1

        accuracy = correct / total
        confusion_MA = confusion_matrix(test_la, pred_labels_la)

        return accuracy, loss, confusion_MA
