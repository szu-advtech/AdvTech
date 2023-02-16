import torch
import torch.nn as nn

import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

class Discriminator(nn.Module):

    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(100, 64),
            nn.LeakyReLU(0.02),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # create loss function
        self.loss_function = nn.MSELoss()

        # 创建优化器
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001)

        # 计数和图像数据存储
        self.counter = 0;
        self.progress = []

        pass

    def forward(self, inputs):
        # simply run model
        return self.model(inputs)

    def train(self, inputs, targets1, targets2):
        # calculate the output of the network
        outputs = self.forward(inputs)

        # calculate loss
        loss = (self.loss_function(outputs, targets1) + self.loss_function(outputs, targets2))/2

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 500 == 0):
            print("counter = ", self.counter)
            pass
        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
        pass

    pass


class Prediction(nn.Module):

    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(100, 64),
            nn.LeakyReLU(0.02),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

        # create loss function
        self.loss_function = nn.MSELoss()

        # 创建优化器
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001)

        # 计数和图像数据存储
        self.counter = 0;
        self.progress = []

        pass

    def forward(self, inputs):
        # simply run model
        return self.model(inputs)

    def train(self, inputs, targets):
        # calculate the output of the network
        outputs = self.forward(inputs)

        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 500 == 0):
            print("counter = ", self.counter)
            pass
        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        pass

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
        pass

    pass

class Generator(nn.Module):

    def __init__(self):
        # initialise parent pytorch class
        super().__init__()

        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.LeakyReLU(0.02),
            nn.Linear(64, 100),
            nn.Sigmoid()
        )

        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.001)

        # counter and accumulator for progress
        self.counter = 0;
        self.progress = []

        pass

    def forward(self, inputs):
        # simply run model
        return self.model(inputs)

    def train(self, P, inputs, targets):
        # calculate the output of the network
        g_output = self.forward(inputs)

        # pass onto Discriminator
        p_output = P.forward(g_output)

        # calculate error
        loss = P.loss_function(p_output, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def train_D(self, D, inputs, targets1, targets2):
        # calculate the output of the network
        g_output = self.forward(inputs)

        # pass onto Discriminator
        d_output = D.forward(g_output)

        # calculate error
        loss = (D.loss_function(d_output, targets1) + D.loss_function(d_output, targets2))/2

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
    pass

    def plot_progress(self):
        df = pd.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16, 8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        plt.show()
        pass

    pass

def creat_dataset(name):
    data = np.array(pd.read_csv(name))
    data_x1 = data[:,0:5]#输出第0-5列
    data_y1 = data[:,5]#输出第6列
    xx, yy = [], []
    print(data_x1.shape)
    data_x1max = np.amax(data_x1, axis=0)
    data_x1min = np.amin(data_x1, axis=0)
    data_y1max = np.amax(data_y1, axis=0)
    data_y1min = np.amin(data_y1, axis=0)
    print(data_y1min)
    for i in range(data_x1.shape[1]):#shape[0]为矩阵行数，shape[1]为矩阵列数
        for j in range(data_x1.shape[0]):
            data_x1[j,i] = (data_x1[j, i] - data_x1min[i]) / (data_x1max[i] - data_x1min[i])
    for j in range(data_y1.shape[0]):
        data_y1[j] = (data_y1[j] - data_y1min) / (data_y1max - data_y1min)
    for i in range(data_x1.shape[0]):#shape[0]为矩阵行数，shape[1]为矩阵列数
        xx.append(data_x1[i, :])
        yy.append(data_y1[i])
    xx = np.array(xx)
    yy = np.array(yy)
    train_x = np.reshape(xx, (xx.shape[0], 1, xx.shape[1])).astype('float32')
    train_x1 = train_x.astype('float32')
    train_x1 = torch.from_numpy(train_x1).to(device)#转换成张量，对数组进行改变时，原数组也会发生变化
    train_y = yy.reshape(-1, 1)
    train_y = np.reshape(yy, (yy.shape[0], 1, 1)).astype('float32')
    train_y1 = train_y.astype('float32')
    train_y1 = torch.from_numpy(train_y1).to(device)
    return train_x1,train_y1


generate_real, real_targets = creat_dataset('dataset11.csv')

D = Discriminator().to(device)
G = Generator().to(device)
P = Prediction().to(device)
for i in range(2000):

    P.train(G.forward(All_data), B5_targets)
    G.train(P, All_data, B5_targets)
    D.train(G.forward(All_data), B5_targets, B6_targets)
    G.train_D(D, All_data, B5_targets, B6_targets)
    # add image to list every 1000
    # if (i % 1000 == 0):
    #     output = g.forward(generate_rand)
    #     print(output)
    pass

predictionB5 = P.forward(G.forward(All_data)).cpu()
predictionB5 = predictionB5.squeeze(2).detach().numpy()
realB5 = B5_targets.cpu()
realB5 = realB5.squeeze(2).detach().numpy()
plt.figure()
plt.plot(predictionB5, 'y*', label='prediction')
plt.plot(realB5, 'b', label="real")
plt.xlabel('cycle')
plt.ylabel('capacity')
plt.show()

predictionB6 = P.forward(G.forward(All_data)).cpu()
predictionB6 = predictionB6.squeeze(2).detach().numpy()
realB6 = B6_targets.cpu()
realB6 = realB6.squeeze(2).detach().numpy()
plt.figure()
plt.plot(predictionB6, 'y*', label='prediction')
plt.plot(realB6, 'b', label="real")
plt.xlabel('cycle')
plt.ylabel('capacity')
plt.show()
# D.plot_progress()
# P.plot_progress()
# G.plot_progress()


n_epoch = 200
for epoch in range(n_epoch):
    for i in range(165):
        p = float(i + epoch * 165) / n_epoch / 165
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
