# 数据归一化处理（标准化）
# 测试数据用

import torch
import pandas as pd
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('device:', device)

def creat_dataset(name):
    data = np.array(pd.read_csv(name))
    data_x1 = data[:,0:4]#输出第0-5列
    data_y1 = data[:,4]#输出第6列
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


generate_real, real_targets = creat_dataset('../dataset11_4features.csv')

print(generate_real)