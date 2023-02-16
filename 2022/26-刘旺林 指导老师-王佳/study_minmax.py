import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.autograd import Function

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

alpha = 0
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class DANNModel(nn.Module):
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_linear1', nn.Linear(4, 64))
        self.feature.add_module('f_layernorm1', nn.LayerNorm(64))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_linear2', nn.Linear(64, 50))
        self.feature.add_module('f_layernorm2', nn.LayerNorm(50))
        # self.feature.add_module('f_drop1', nn.Dropout(0.2))
        self.feature.add_module('f_sig', nn.Sigmoid())

        self.lstm = nn.LSTM(input_size = 50, hidden_size = 128, num_layers = 1)
        self.lstm.layernorm = nn.LayerNorm(128)
        self.lstm.relu = nn.ReLU(True)
        self.lstm.drop = nn.Dropout(0.2)
        self.lstm.linear1 = nn.Linear(128, 1)
        # self.lstm.leakyrelu = nn.LeakyReLU(0.02)
        # self.lstm.linear2 = nn.Linear(128, 1)
        self.lstm.sig = nn.Sigmoid()

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_linear1', nn.Linear(50, 100))
        self.domain_classifier.add_module('d_layernorm1', nn.LayerNorm(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_linear2', nn.Linear(100, 1))
        self.domain_classifier.add_module('d_sig', nn.Sigmoid())

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        feature, _ = self.lstm(feature)
        self.lstm.layernorm
        self.lstm.relu
        self.lstm.drop
        feature = self.lstm.linear1(feature)
        self.lstm.sig
        class_output = feature
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output


def creat_dataset(name):
    data = np.array(pd.read_csv(name))
    data_x1 = data[:,0:4]#输出第0-5列
    data_y1 = data[:,4]#输出第6列
    xx, yy = [], []
    data_x1max = np.amax(data_x1, axis=0)
    data_x1min = np.amin(data_x1, axis=0)
    data_y1max = np.amax(data_y1, axis=0)
    print(data_y1max)
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

def remake_dataset(targets, name):
    data = np.array(pd.read_csv(name))
    data_x1 = data[:,0:4]#输出第0-5列
    data_y1 = data[:,4]#输出第6列
    print(data_y1.shape)
    yy = []
    data_y1max = np.amax(data_y1, axis=0)
    print(data_y1max)
    data_y1min = np.amin(data_y1, axis=0)
    print(data_y1min)
    for j in range(data_y1.shape[0]):
        data_y1[j] = (data_y1max - data_y1min) * targets[j] + data_y1min
    for i in range(data_y1.shape[0]):#shape[0]为矩阵行数，shape[1]为矩阵列数
        yy.append(data_y1[i])
    yy = np.array(yy)
    train_y = yy.reshape(-1, 1)
    train_y = np.reshape(yy, (yy.shape[0], 1, 1)).astype('float32')
    train_y1 = train_y.astype('float32')
    train_y1 = torch.from_numpy(train_y1).to(device)
    return train_y1

B5_features, B5_targets = creat_dataset('../dataset11_4features.csv')
B6_features, B6_targets =creat_dataset('../dataset22_4features.csv')
B7_features, B7_targets =creat_dataset('../dataset33_4features.csv')

model = torch.load('DANN_model_best.pt')

model.eval()
predictionB5, _ = model(B5_features, alpha = alpha)
predictionB6, _ = model(B6_features, alpha = alpha)
predictionB7, _ = model(B7_features, alpha = alpha)
# print(predictionB5)
# print(predictionB6)
predictionB5 = remake_dataset(predictionB5, '../dataset11_4features.csv')
real_B5 = remake_dataset(B5_targets, '../dataset11_4features.csv')
predictionB6 = remake_dataset(predictionB6, '../dataset22_4features.csv')
real_B6 = remake_dataset(B6_targets, '../dataset22_4features.csv')
predictionB7 = remake_dataset(predictionB7, '../dataset33_4features.csv')
real_B7 = remake_dataset(B7_targets, '../dataset33_4features.csv')
# print(predictionB5)
# print(real_B5)
# print(predictionB6)
# print(real_B6)
predictionB5 = predictionB5.cpu()
predictionB5 = predictionB5.squeeze(2).detach().numpy()
realB5 = real_B5.cpu()
realB5 = realB5.squeeze(2).detach().numpy()

predictionB6 = predictionB6.cpu()
predictionB6 = predictionB6.squeeze(2).detach().numpy()
realB6 = real_B6.cpu()
realB6 = realB6.squeeze(2).detach().numpy()

predictionB7 = predictionB7.cpu()
predictionB7 = predictionB7.squeeze(2).detach().numpy()
realB7 = real_B7.cpu()
realB7 = realB7.squeeze(2).detach().numpy()

plt.figure(figsize=(9, 8), dpi=80)
plt.figure(1)

ax1 = plt.subplot(221)
ax1.plot(predictionB5, 'r', label='prediction')
ax1.plot(realB5, 'b', label="real")
ax1.set_xlabel('cycle')
ax1.set_ylabel('B5_capacity')

ax2 = plt.subplot(222)
ax2.plot(predictionB6, 'r', label='prediction')
ax2.plot(realB6, 'b', label="real")
ax2.set_xlabel('cycle')
ax2.set_ylabel('B6_capacity')

ax3 = plt.subplot(223)
ax3.plot(predictionB7, 'r', label='prediction')
ax3.plot(realB7, 'b', label="real")
ax3.set_xlabel('cycle')
ax3.set_ylabel('B7_capacity')
plt.show()