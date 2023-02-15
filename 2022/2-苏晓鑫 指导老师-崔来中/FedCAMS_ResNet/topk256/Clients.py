from torchvision import datasets, transforms, models
import torch
from torch import nn
import Model as m
import numpy as np
import time
import heapq
import math
import random
import os
import scipy.stats as st
from scipy.optimize import curve_fit

myseed = 0
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
os.environ['PYTHONHASHSEED'] = str(myseed)
random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)

batch_size = 20
epoch_num = 3

transform_train = transforms.Compose([
    # transforms.RandomCrop(32, padding=4),  # 先四周填充0，在吧图像随机裁剪成32*32
    # transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    # transforms.Normalize((0.49139968, 0.48215841, 0.44653091), (0.24703223, 0.24348513, 0.26158784)),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


class Client():
    def __init__(self, client_num):
        self.client_num = client_num
        self.model = torch.load("../../init_resnet.pkl")  # m.ResNet18()  # m.ResNet(m.ResidualBlock)
        self.last_model = torch.load("../../init_resnet.pkl")  # m.ResNet18()
        print(self.model)
        train_data = datasets.CIFAR10('../../../CIFAR-10', train=True, download=True,
                                      transform=transform_train)
        self.lr = 0.01
        self.data_len = []
        self.communication_time = [0] * self.client_num
        self.train_loader = []

        total_trainable_params = 0
        for p in self.model.parameters():
            total_trainable_params += p.numel()
        print("total parameter num is ", total_trainable_params)
        self.proportion = 1
        self.d = total_trainable_params
        self.traffic = self.d * self.proportion * (32 + int(math.log(self.d, 2)) + 1) / 8 / 1024 / 1024

        self.accu_gra = torch.zeros((client_num, total_trainable_params))

        print(range(len(train_data)))
        data_len = int(50000 / self.client_num)
        print("data len is ", data_len)

        # index_list = list(range(len(train_data)))
        # random.shuffle(index_list)

        for i in range(self.client_num):
            self.data_len.append(data_len)
            start_index = int(len(train_data) / self.client_num) * i
            print("the start index is ", start_index)
            data = []
            for j in range(data_len):
                idx = (start_index + j) % len(train_data)
                # data.append(train_data[index_list[idx]])
                data.append(train_data[idx])
            train_loader = torch.utils.data.DataLoader(
                data,
                batch_size=batch_size, shuffle=True
            )
            self.train_loader.append(train_loader)
            print(i)

    def get_dataLen(self, idx):
        return self.data_len[idx]

    def change_lr(self, learning_rate):
        self.lr = [learning_rate] * self.client_num
        print("now lr is ", self.lr)

    def get_model(self):
        return self.model

    def get_model_gradient(self, idx):
        parameters = torch.tensor([])
        num_batches_tracked = 0
        for p in self.model.parameters():
            parameters = torch.cat((parameters, p.view(-1).detach()))
        last_parameters = torch.tensor([])
        for p in self.last_model.parameters():
            last_parameters = torch.cat((last_parameters, p.view(-1).detach()))
        self.accu_gra[idx] = parameters - last_parameters - self.accu_gra[idx]

        proportion = 1 / 256
        value, indices = torch.topk(self.accu_gra[idx].abs(), int(self.accu_gra[idx].size()[0] * proportion))
        print("top {}% is {}".format(proportion * 100, torch.min(value)))

        new_gra = torch.zeros(self.accu_gra[idx].shape)
        new_gra[indices] = self.accu_gra[idx][indices]
        self.accu_gra[idx][indices] = 0
        return new_gra, num_batches_tracked

    def set_model(self, model):
        self.model.load_state_dict(model.state_dict())
        self.last_model.load_state_dict(model.state_dict())
        # torch.save(model, "net_params.pkl")
        # self.model = torch.load("net_params.pkl")
        # self.last_model = torch.load("net_params.pkl")

    def adjust_learning_rate(self):
        self.lr *= 0.5
        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = param_group['lr'] * decay_rate
        #     self.lr = param_group['lr']

    def train(self, idx, communication_time):
        self.communication_time[idx] += 1
        print("client no is ", idx, " train num is ", self.communication_time[idx])
        self.model = self.model.to(device)
        loss_fn = torch.nn.CrossEntropyLoss().to(device)

        lr = self.lr
        print("learning rate is ", lr)
        # optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-6,
        #                             nesterov=True)

        optimizer = torch.optim.SGD(params=self.model.parameters(), lr=lr)
        for epo in range(epoch_num):
            self.model.train()
            for i, (data, label) in enumerate(self.train_loader[idx]):
                data = data.to(device)
                label = label.to(device)

                optimizer.zero_grad()

                pred = self.model(data)
                loss = loss_fn(pred, label)
                loss.backward()
                optimizer.step()

                if i % 100 == 0:
                    print("Train Epoch: {}, iteration: {}, Loss: {}".format
                          (epo, i, loss.item()))

        self.model = self.model.to('cpu')
