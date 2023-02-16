import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import random
import numpy as np
from torch.autograd import Function
from sklearn.preprocessing import MinMaxScaler



data = np.array(pd.read_csv('../dataset22_4features.csv'))
data_x1 = data[:,0:4]#输出第0-5列
data_y1 = data[:,4]#输出第6列
xx, yy = [], []
scaler = MinMaxScaler()
data_y1 = scaler.fit_transform(data_y1.reshape(-1,1))
print(data_y1.shape)
for i in range(data_x1.shape[0]):  # shape[0]为矩阵行数，shape[1]为矩阵列数
    yy.append(data_y1[i])
yy = np.array(yy)
train_y = yy.reshape(-1, 1)
train_y = np.reshape(yy, (yy.shape[0], 1, 1)).astype('float32')
train_y1 = train_y.astype('float32')
train_y1 = torch.from_numpy(train_y1)
# print(train_y1)
print(train_y1.shape)
train_y1 = train_y1.squeeze(2).numpy()
train_y1 = scaler.inverse_transform(train_y1)
# print(train_y1)