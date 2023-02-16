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
        # self.feature.add_module('f_bn1', nn.BatchNorm1d(64))
        self.feature.add_module('f_relu1', nn.LeakyReLU(0.02))
        self.feature.add_module('f_layernorm', nn.LayerNorm(64))
        self.feature.add_module('f_linear2', nn.Linear(64, 50))
        self.feature.add_module('f_sig', nn.Sigmoid())

        self.lstm = nn.LSTM(input_size = 50, hidden_size = 256, num_layers = 1)
        self.lstm.relu = nn.LeakyReLU(0.02)
        self.lstm.layernorm = nn.LayerNorm(256)
        self.lstm.linear1 = nn.Linear(256, 1)
        # self.lstm.leakyrelu = nn.LeakyReLU(0.02)
        # self.lstm.linear2 = nn.Linear(128, 1)
        self.lstm.sig = nn.Sigmoid()

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_linear1', nn.Linear(50, 64))
        # self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(64))
        self.domain_classifier.add_module('d_relu1', nn.LeakyReLU(0.02))
        self.domain_classifier.add_module('d_layernorm', nn.LayerNorm(64))
        self.domain_classifier.add_module('d_linear2', nn.Linear(64, 2))
        self.domain_classifier.add_module('d_sig', nn.Sigmoid())

    def forward(self, input_data, alpha):
        feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        feature, _ = self.lstm(feature)
        self.lstm.relu
        self.lstm.layernorm
        feature = self.lstm.linear1(feature)
        # self.lstm.leakyrelu
        # feature = self.lstm.linear2(feature)
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
    data_y1min = np.amin(data_y1, axis=0)
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


B5_source, B5_source_targets = creat_dataset('../dataset11_4features.csv')
B6_target, B6_target_targets =creat_dataset('../dataset22_4features.csv')

cuda = True

my_net = DANNModel()

optimizer = optim.Adam(my_net.parameters(), lr=0.0001)

loss_class = torch.nn.MSELoss()
loss_domain = torch.nn.MSELoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True
batch_size = 2
n_epoch = 60
xx = np.zeros([165, 1])

prev_loss = 10
for epoch in range(n_epoch):
    optimizer.zero_grad()
    for i in range(165):
        x = 0
        p = float(i + epoch * 165) / n_epoch / 165
        alpha = 2. / (1. + np.exp(-10 * p)) - 1
        data_source = B5_source[i]
        s_img = data_source
        s_label = B5_source_targets[i]
        domain_label = torch.zeros([1, 2])
        if cuda:
            s_img = s_img.cuda()
            s_label = s_label.cuda()
            domain_label = domain_label.cuda()
        class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
        xx[i] = class_output.cpu().detach().numpy()
        err_s_label = loss_class(class_output, s_label)
        err_s_domain = loss_domain(domain_output, domain_label)
        data_target = B6_target[i]
        t_img = data_target
        domain_label = torch.ones([1, 2])
        if cuda:
            t_img = t_img.cuda()
            domain_label = domain_label.cuda()

        _, domain_output = my_net(input_data=t_img, alpha=alpha)
        err_t_domain = loss_domain(domain_output, domain_label)
        err = err_t_domain + err_s_domain + err_s_label

        err.backward()
        optimizer.step()
        x = x + err_s_label.item()
        if (i % 10 == 0):
            print('原数据预测模型loss：',err_s_label.item(),'===','原数据鉴别模型loss：',err_s_domain.item())
            print('目标数据鉴别模型loss：',err_t_domain.item())
    loss = x / 165
    print(loss)
    if loss < prev_loss:
        torch.save(my_net, 'DANN_model.pt')  # save model parameters to files
        prev_loss = loss
    print('训练进度:',(epoch/n_epoch) * 100,'%')

predictionB5 = xx
realB5 = B5_source_targets.cpu()
realB5 = realB5.squeeze(2).detach().numpy()
plt.figure()
plt.plot(predictionB5, 'r', label='prediction')
plt.plot(realB5, 'b', label="real")
plt.xlabel('cycle')
plt.ylabel('capacity')
plt.show()

for i in range(165):
    p = float(i + epoch * 165) / n_epoch / 165
    alpha = 2. / (1. + np.exp(-10 * p)) - 1
    data_source = B6_target[i]
    s_img = data_source
    s_label = B6_target_targets[i]
    domain_label = torch.zeros([1, 2])
    if cuda:
        s_img = s_img.cuda()
        s_label = s_label.cuda()
        domain_label = domain_label.cuda()
    class_output, domain_output = my_net(input_data=s_img, alpha=alpha)
    xx[i] = class_output.cpu().detach().numpy()

predictionB6 = xx
realB6 = B6_target_targets.cpu()
realB6 = realB6.squeeze(2).detach().numpy()
plt.figure()
plt.plot(predictionB6, 'r', label='prediction')
plt.plot(realB6, 'b', label="real")
plt.xlabel('cycle')
plt.ylabel('capacity')
plt.show()