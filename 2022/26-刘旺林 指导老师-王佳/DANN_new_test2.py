import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.autograd import Function
from DANN_model import test

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
        self.feature.add_module('f_layernorm1', nn.LayerNorm(64))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_linear2', nn.Linear(64, 50))
        self.feature.add_module('f_layernorm2', nn.LayerNorm(50))
        self.feature.add_module('f_sig', nn.Sigmoid())

        # self.lstm = nn.LSTM(input_size = 50, hidden_size = 6, num_layers = 1)
        # self.lstm.layernorm = nn.LayerNorm(6)
        # self.lstm.relu = nn.ReLU(True)
        # self.lstm.drop = nn.Dropout(0.2)
        # self.lstm.linear1 = nn.Linear(6, 1)
        # # self.lstm.leakyrelu = nn.LeakyReLU(0.02)
        # # self.lstm.linear2 = nn.Linear(128, 1)
        # self.lstm.sig = nn.Sigmoid()

        self.label_predictor = nn.LSTM(input_size = 50, hidden_size = 6, num_layers = 1)
        self.label_predictor.add_module('l_layernorm1', nn.LayerNorm(6))
        self.label_predictor.add_module('l_relu1', nn.ReLU(True))
        self.label_predictor.add_module('l_drop1', nn.Dropout(0.2))
        self.label_predictor.add_module('l_linear1', nn.Linear(6, 1))
        self.label_predictor.add_module('l_sig', nn.Sigmoid())

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
        label_output = feature
        domain_output = self.domain_classifier(reverse_feature)

        return label_output, domain_output

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
LR = 0.05
optimizer = optim.Adam(my_net.parameters(), lr=LR)

loss_class = torch.nn.MSELoss()
loss_domain = torch.nn.MSELoss()

if cuda:
    my_net = my_net.cuda()
    loss_class = loss_class.cuda()
    loss_domain = loss_domain.cuda()

for p in my_net.parameters():
    p.requires_grad = True
batch_size = 2
n_epoch = 20000
xx = np.zeros([165, 1])

best_loss = 10
for epoch in range(n_epoch):

    x = 0
    p1 = float(epoch) / n_epoch
    alpha = 2. / (1. + np.exp(-10 * p1)) - 1
    #随着迭代次数改变学习率
    # for p in optimizer.param_groups:
    #     p['lr'] = LR/((1 + 3 * p1 )**0.1)
    data_source = B5_source
    s_img = data_source
    s_label = B5_source_targets
    domain_label = torch.zeros([165, 1, 1])
    if cuda:
        s_img = s_img.cuda()
        s_label = s_label.cuda()
        domain_label = domain_label.cuda()
    label_output, domain_output = my_net(input_data=s_img, alpha=alpha)
    # xx[i] = class_output.cpu().detach().numpy()
    err_s_label = loss_class(label_output, s_label)
    err_s_domain = loss_domain(domain_output, domain_label)
    data_target = B6_target
    t_img = data_target
    domain_label = torch.ones([165, 1, 1])
    if cuda:
        t_img = t_img.cuda()
        domain_label = domain_label.cuda()

    _, domain_output = my_net(input_data=t_img, alpha=alpha)
    err_t_domain = loss_domain(domain_output, domain_label)
    err = err_t_domain + err_s_domain + err_s_label

    optimizer.zero_grad()
    err.backward()
    optimizer.step()

    if (epoch % 100 == 0):
        print('原数据预测模型loss：{:.5f}，原数据鉴别模型loss：{:.5f}，目标数据鉴别模型loss：{:.5f}'.format(err_s_label.item(), err_s_domain.item(), err_t_domain.item()))
        print('训练进度：{:.2f}%'.format((epoch/n_epoch) * 100))

    torch.save(my_net, 'DANN_model.pt')

    loss_B6 = test(B6_target, B6_target_targets)
    loss_B5 = test(B5_source, B5_source_targets)
    if best_loss > (loss_B5.item() + loss_B6.item()) / 2:
        best_loss = (loss_B5.item()+loss_B6.item()) / 2
        print(best_loss)
        torch.save(my_net,'DANN_model_best_1.pt')
        print('更新模型ing')

print(best_loss)