import scipy.io as sio
import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from torch.autograd import Variable
import math
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

# Define LSTM Neural Networks
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
        # s, b, h = x.shape  # x is output, size (seq_len, batch, hidden_size)
        # x = x.view(s * b, h)
        x = self.linear1(x)
        # x = x.view(s, b, -1)

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


train1_x, train1_y = creat_dataset('../dataset11_4features.csv')
#test1_x, test1_y = creat_dataset("dataset2.csv")


if __name__ == '__main__':
    # ----------------- train -------------------
    INPUT_FEATURES_NUM = 4
    OUTPUT_FEATURES_NUM = 1


    lstm_model = LstmRNN(INPUT_FEATURES_NUM, 128, output_size=OUTPUT_FEATURES_NUM, num_layers=1).to(device)  # 20 hidden units
    print('LSTM model:', lstm_model)
    print('model.parameters:', lstm_model.parameters)
    print('train x tensor dimension:', Variable(train1_x).size())

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(lstm_model.parameters(), lr=0.001)

    prev_loss = 1000
    max_epochs = 3000

    train_x_tensor = train1_x.to(device)

    for epoch in range(max_epochs):
        output = lstm_model(train_x_tensor).to(device)
        loss = criterion(output, train1_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if loss < prev_loss:
            torch.save(lstm_model, 'lstm1_model.pt')  # save model parameters to files
            prev_loss = loss

        if loss.item() < 1e-8:
            print('Epoch [{}/{}], Loss: {:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            print("The loss value is reached")
            break
        elif (epoch + 1) % 100 == 0:
            print('Epoch: [{}/{}], Loss:{:.5f}'.format(epoch + 1, max_epochs, loss.item()))
            #print(output)

    predictionB5 = lstm_model(train1_x).cpu()
    predictionB5 = predictionB5.squeeze(2).detach().numpy()
    realB5 = train1_y.cpu()
    realB5 = realB5.squeeze(2).detach().numpy()
    plt.figure()
    plt.plot(predictionB5, 'r', label='prediction')
    plt.plot(realB5, 'b', label="real")
    plt.xlabel('cycle')
    plt.ylabel('capacity')
    plt.show()
