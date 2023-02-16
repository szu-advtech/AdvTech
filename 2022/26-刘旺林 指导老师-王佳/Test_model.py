import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)


class LstmRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)  # utilize the LSTM model in torch.nn
        self.linear1 = nn.Linear(hidden_size, output_size)  # 全连接层

    def forward(self, _x):
        x, _ = self.lstm(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = x.view(s, b, -1)
        return x

class GruRNN(nn.Module):
    """
        Parameters：
        - input_size: feature size
        - hidden_size: number of hidden units
        - output_size: number of output
        - num_layers: layers of LSTM to stack
    """

    def __init__(self, input_size, hidden_size=1, output_size=1, num_layers=1):
        super().__init__()

        self.gru = nn.GRU(input_size, hidden_size, num_layers)  # utilize the GRU model in torch.nn
        self.linear1 = nn.Linear(hidden_size, 16)  # 全连接层
        self.linear2 = nn.Linear(16, output_size)  # 全连接层

    def forward(self, _x):
        x, _ = self.gru(_x)  # _x is input, size (seq_len, batch, input_size)
        s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        x = x.view(s * b, h)
        x = self.linear1(x)
        x = self.linear2(x)
        x = x.view(s, b, -1)
        return x

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






B5_features, B5_targets = creat_dataset('../dataset11_4features.csv')
B6_features, B6_targets =creat_dataset('../dataset22_4features.csv')
B7_features, B7_targets =creat_dataset('../dataset33_4features.csv')

model = torch.load('lstm1_model.pt')

model.eval()

predictionB5 = model(B5_features)

predictionB5 = predictionB5.cpu()
predictionB5 = predictionB5.squeeze(2).detach().numpy()
realB5 = B5_targets.cpu()
realB5 = realB5.squeeze(2).detach().numpy()

predictionB6 = model(B6_features)

predictionB6 = predictionB6.cpu()
predictionB6 = predictionB6.squeeze(2).detach().numpy()
realB6 = B6_targets.cpu()
realB6 = realB6.squeeze(2).detach().numpy()

predictionB7 = model(B7_features)

predictionB7 = predictionB7.cpu()
predictionB7 = predictionB7.squeeze(2).detach().numpy()
realB7 = B7_targets.cpu()
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