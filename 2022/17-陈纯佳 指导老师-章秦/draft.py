# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 20:49:32 2022

@author: C2J
"""

import numpy as np
from scipy.stats import expon
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import gamma
import math as math
import random
from sklearn.model_selection import KFold

#ED的PDF的真实图像
fig,ax = plt.subplots(1,1)
lambdaUse = 0.5
loc = 0
scale = 1.0/lambdaUse
#平均值、方差、偏度、峰度
mean,var,skew,kurt = expon.stats(loc,scale,moments='mvsk')
#ppf:累积分布函数的反函数
x = np.linspace(expon.ppf(0.01,loc,scale),
                expon.ppf(0.99,loc,scale),100)
#ax.plot(x,expon.pdf(x,loc,scale))
#plt.title(u'ED')
#plt.show()

#ND的PDF的真实图像
#样本的x和y的值
#数据集应该包含y，作为后续误差计算的F
#计算对应theta的PDF
def norm_dist_calmu(theta,data):
    y = norm.cdf(theta, loc=np.mean(data), scale=np.std(data))
    return y
def norm_dist_prob(theta,data):
    y = norm.pdf(theta, loc=np.mean(data), scale=np.std(data))
    return y
# 均值0.5，方差1,正态分布模拟数据50个
data = np.random.normal(0.5, 1, 50)
x1 = np.linspace(norm.ppf(0.4,loc=np.mean(data), scale=np.std(data)),
                 norm.ppf(0.99,loc=np.mean(data), scale=np.std(data)), 
                 len(data)) 
F1 = norm_dist_calmu(x1 ,data)
P1 = norm_dist_prob(x1, data)
#plt.plot(x1, P1,'g',label='pdf')
#plt.title(u'ND,a=0.5,b=1')

#GD的PDF的真实图像
x = np.arange(0.01,10,0.25)
#print(x)
y = gamma.pdf(x,1,scale=2)
#print(y)
f = gamma.cdf(x,1,scale=2)
#print(f)
#plt.plot(x, y,'g',label='ND')
#plt.xlabel('GD,a=1,b=2')
#plt.ylabel('PDF')

eta = 0.1
#输入层、隐藏层、输出层的维度
input_dim = 1
hidden_dim = 3
output_dim = 1
#隐藏层到输出层的权重，随机生成
synapse_list = []
for i in range(hidden_dim):
    data1 = random.random()
    synapse_list.append(data1)
synapse = np.array(synapse_list)
print(synapse)
#指数分布
def expon_dist_calmu(data,lambda_):
    y = expon.cdf(data,scale=1/lambda_)
    return y
def expon_dist_prob(data,lambda_):
    y = expon.pdf(data,scale=1/lambda_)
    return y
x1 = np.linspace(expon.ppf(0.01,loc=0, scale=1/0.5),expon.ppf(0.99,loc=0, scale=1/0.5), 50) 
F1 = expon_dist_calmu(x1, 0.5)
P1 = expon_dist_prob(x1, 0.5)
#正态分布
def norm_dist_calmu(theta,data):
    y = norm.cdf(theta, loc=np.mean(data), scale=np.std(data))
    return y
def norm_dist_prob(theta,data):
    y = norm.pdf(theta, loc=np.mean(data), scale=np.std(data))
    return y
data_ND = np.random.normal(0.5, 1, 50)
x2 = np.linspace(norm.ppf(0.6,loc=np.mean(data_ND), scale=np.std(data_ND)),
                 norm.ppf(0.99,loc=np.mean(data_ND), scale=np.std(data_ND)), 
                 len(data_ND)) 
F2 = norm_dist_calmu(x2 ,data_ND)
P2 = norm_dist_prob(x2, data_ND)
#伽马分布
x3 = np.arange(0.01,10,0.25)
F3 = gamma.cdf(x3,1,scale=2)
P3 = gamma.pdf(x3,1,scale=2)
#print(P4)
def sigmoid(x):
    output = 1/(1+np.exp(-x))
    return output
def sigmoid_output_to_detivative(output):
    return output*(1-output)
alpha = 0.1
M_f = 0
M_p = 0
kf = KFold(n_splits = 5, shuffle=True, random_state=0)    # 5折
for train_index, test_index in kf.split(x3,F3,P3):     # 将数据划分为k折
    train_x = x3[train_index]   # 选取的训练集数据下标
    train_F = F3[train_index]
    train_P = P3[train_index]
    test_x = x3[test_index]     # 选取的测试集数据下标
    test_F = F3[test_index]
    test_P = P3[test_index]
    #五折交叉验证的训练部分
    for k in range(300):
        CDF = []
        PDF = []
        for index in range(len(train_x)):
            #print(index)
            x = train_x[index]
            F = train_F[index]
            #print(x)
            #根据数据集中的x的值得到隐藏层每个激活函数的输出，一个一维数组
            hidden_layer1 = sigmoid(x)
            hidden_layer2 = sigmoid(x)
            hidden_layer3 = sigmoid(x)
            hidden_layer = np.array([hidden_layer1,hidden_layer2,hidden_layer3])
            #print(hidden_layer)
            #neural network final output 计算权重之和
            synapse_sum = np.sum(np.fromiter(synapse, float))
            #print(synapse_sum)
            #点乘计算神经网络的输出
            F_ = (np.dot(hidden_layer, synapse))/(synapse_sum)
            #print(F_)
            #use the estimate of F to estimate 利用论文中的公式，计算根据神经网络的输出F_对x进行微分得到的PDF的值
            P_ = synapse[0]*sigmoid_output_to_detivative(hidden_layer1) + synapse[1]*sigmoid_output_to_detivative(hidden_layer2) + synapse[2]*sigmoid_output_to_detivative(hidden_layer3)
            #print(P_)
            CDF.append(F_)
            PDF.append(P_)
            #F is actual data,F_ is practical data
            #定义输出误差
            output_delta = F_ - F
            output_error = (output_delta**2)/2
            synapse_update = output_delta*(np.dot(hidden_layer, synapse)/((synapse_sum)**2))
            synapse = synapse - eta*synapse_update
    error_F = 0
    error_P = 0
    for i in range(len(test_x)):
        x = test_x[i]
        F = test_F[i]
        P = test_P[i]
        hidden_layer1 = sigmoid(x)
        hidden_layer2 = sigmoid(x)
        hidden_layer3 = sigmoid(x)
        hidden_layer = np.array([hidden_layer1,hidden_layer2,hidden_layer3])
        synapse_sum = np.sum(np.fromiter(synapse, float))
        F_ = (np.dot(hidden_layer, synapse))/(synapse_sum)
        P_ = synapse[0]*sigmoid_output_to_detivative(hidden_layer1) + synapse[1]*sigmoid_output_to_detivative(hidden_layer2) + synapse[2]*sigmoid_output_to_detivative(hidden_layer3)
        error_F = error_F + (abs(F - F_))/F
        error_P = error_F + (abs(P - P_))/P
    M_f = M_f + error_F/len(test_x)/100
    M_p = M_p + error_P/len(test_x)/100
print(M_f)
print(M_p)
plt.plot(train_x, PDF,'g',label='pdf')
plt.xlabel('GD,a=1,b=2')
plt.ylabel('PDF')




























