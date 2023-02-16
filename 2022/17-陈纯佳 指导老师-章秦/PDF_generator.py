# -*- coding: utf-8 -*-
"""
Created on Thu Dec  1 22:56:25 2022

@author: C2J
"""
import numpy as np
import math as math
from scipy.stats import norm
from scipy.stats import expon
from scipy.stats import gamma
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
#定义学习率为0.1
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
# 样本的x和y的值，数据集应该包含y，作为后续误差计算的F
# 数组x所对应的PDF的数组F
# linspace() 函数返回指定间隔内均匀间隔数字的 ndarray。
# scipy.stats.norm.ppf(0.95，loc=0，scale=1)返回累积分布函数中概率等于0.95对应的x值（CDF函数中已知y求对应的x）。
#指数分布
def expon_dist_calmu(data,lambda_):
    y = expon.cdf(data,scale=1/lambda_)
    return y
def expon_dist_prob(data,lambda_):
    y = expon.pdf(data,scale=1/lambda_)
    return y
x1 = np.linspace(expon.ppf(0.01,loc=0, scale=1/0.5),expon.ppf(0.99,loc=0, scale=1/0.5), 40) 
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
#data_LD = np.random.lognormal(0.5, 1, 50)
#x3 = np.linspace(norm.ppf(0.3,loc=np.mean(data_LD), scale=np.std(data_LD)),norm.ppf(0.99,loc=np.mean(data_LD), scale=np.std(data_LD)), len(data_LD)) 
#F3 = norm_dist_calmu(x3 ,data_LD)
#P3 = norm_dist_prob(x3, data_LD)

#伽马分布
x3 = np.arange(0.01,10,0.25)
F3 = gamma.cdf(x3,1,scale=2)
P3 = gamma.pdf(x3,1,scale=2)
print(x3)
print(F3)
print(P3)
#dataset = [list(item) for item in zip(x1,F1)]
#print(dataset)
#初始化每个参数
#ED parameter 指数分布的概率分布函数作为激活函数，需要学习参数lambda
lambda_ = 0.5
#ND parameter 正态分布的概率分布函数作为激活函数，需要学习参数mu1和s1
mu_1 = 2
s_1 = 0.5
#LD parameter 对数正态分布的概率分布函数作为激活函数，需要学习参数mu2和s2
mu_2 = 1
s_2 = 0.5
#define four activate functions
#定义三个激活函数
def f1_ED(x,lambda_):
    return 1 - np.exp(-x * lambda_)

def f2_ND(x,mu_1,s_1):
    layer_2 = (1 + math.erf((x-mu_1)/(math.sqrt(2)*s_1)))/2
    return layer_2

def f3_LD(x,mu_2,s_2):
    layer_3 = (1 + math.erf((math.log(x,math.e)-mu_2)/(math.sqrt(2)*s_2)))/2
    return layer_3

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
            hidden_layer1 = f1_ED(x,lambda_)
            hidden_layer2 = f2_ND(x,mu_1,s_1)
            hidden_layer3 = f3_LD(x,mu_2,s_2)
            hidden_layer = np.array([hidden_layer1,hidden_layer2,hidden_layer3])
            #print(hidden_layer)
            #neural network final output 计算权重之和
            synapse_sum = np.sum(np.fromiter(synapse, float))
            #print(synapse_sum)
            #点乘计算神经网络的输出
            F_ = (np.dot(hidden_layer, synapse))/(synapse_sum)
            #print(F_)
            #use the estimate of F to estimate 利用论文中的公式，计算根据神经网络的输出F_对x进行微分得到的PDF的值
            P_ = synapse[0]*(lambda_*np.exp(-lambda_*x)) + synapse[1]*(1/(math.sqrt(2*math.pi*(s_1**2))))*(np.exp(-((x-mu_1)**2)/(2*(s_1**2)))) 
            + synapse[2]*(1/(x*s_2*math.sqrt(2*math.pi)))*(np.exp(-(math.log(x,math.e)-mu_2)**2/(2*(s_2**2))))
            #print(P_)
            CDF.append(F_)
            PDF.append(P_)
            #F is actual data,F_ is practical data
            #定义输出误差
            output_delta = F_ - F
            output_error = (output_delta**2)/2
            #calculate the gradient of all parameters
            #the gradient of weight 根据论文权重更新的公式，计算梯度
            list1=[ 0, f1_ED(x,lambda_)-f2_ND(x,mu_1,s_1), f1_ED(x,lambda_)-f3_LD(x,mu_2,s_2),
                   f2_ND(x,mu_1,s_1)-f1_ED(x,lambda_), 0, f2_ND(x,mu_1,s_1)-f3_LD(x,mu_2,s_2),
                   f3_LD(x,mu_2,s_2)-f1_ED(x,lambda_), f3_LD(x,mu_2,s_2)-f2_ND(x,mu_1,s_1), 0]
            #将list转化为3行3列格式的二维数组
            ans=[]
            for i in range(hidden_dim):
                temp = []
                for j in range(hidden_dim):
                    temp.append(list1[i*hidden_dim+j])
                ans.append(temp)
                    #测试输出
            ans_ = np.array(ans)
            #print("ans_=",ans_)
            #根据论文公式，计算权重的梯度
            synapse_update = output_delta*(np.dot(ans, synapse)/((synapse_sum)**2))
            #print(synapse_update)
            #存放各个激活函数参数的梯度
            #the gradient of lambda_
            lambda_update = output_delta*(synapse[0]/synapse_sum)*(x*np.exp(-lambda_*x))
            #the gradient of mu_1
            mu_1_update = output_delta*(synapse[1]/synapse_sum)*((-1/(math.sqrt(2*math.pi*(s_1**2))))*(np.exp(-((x-mu_1)**2)/(2*(s_1**2)))))
            #the gradient of s_1
            s_1_update = output_delta*(synapse[1]/synapse_sum)*((-(x-mu_1)/(math.sqrt(2*math.pi)*(s_1**2)))*(np.exp(-((x-mu_1)**2)/(2*(s_1**2)))))
            #the gradient of mu_2
            mu_2_update = output_delta*(synapse[2]/synapse_sum)*((-1/(math.sqrt(2*math.pi*(s_2**2))))*(np.exp(-(math.log(x,math.e)-mu_2)**2/(2*(s_2**2)))))
            #the gradient of s_2
            s_2_update = output_delta*(synapse[2]/synapse_sum)*(((-(math.log(x,math.e)-mu_2))/(math.sqrt(2*math.pi)*(s_2**2)))*(np.exp(-(math.log(x,math.e)-mu_2)**2/(2*(s_2**2)))))
            #the gradient of alpha_
            #the gradient of beta_
            #update all parameters of activate functions
            #所有参数梯度下降一次后的结果，eta学习率
            synapse = synapse - eta*synapse_update
            lambda_ = lambda_ - eta*lambda_update
            mu_1 = mu_1 - eta*mu_1_update
            s_1 = s_1 - eta*s_1_update
            mu_2 = mu_2 - eta*mu_2_update
            s_2 = s_2 - eta*s_2_update
            #记录数据经过神经网络后计算得出的CDF和PDF的值
            #五折交叉验证的验证部分
    error_F = 0
    error_P = 0
    for i in range(len(test_x)):
        x = test_x[i]
        F = test_F[i]
        P = test_P[i]
        hidden_layer1 = f1_ED(x,lambda_)
        hidden_layer2 = f2_ND(x,mu_1,s_1)
        hidden_layer3 = f3_LD(x,mu_2,s_2)
        hidden_layer = np.array([hidden_layer1,hidden_layer2,hidden_layer3])
        synapse_sum = np.sum(np.fromiter(synapse, float))
        F_ = (np.dot(hidden_layer, synapse))/(synapse_sum)
        P_ = synapse[0]*(lambda_*np.exp(-lambda_*x)) 
        + synapse[1]*(1/(math.sqrt(2*math.pi*(s_1**2))))*(np.exp(-((x-mu_1)**2)/(2*(s_1**2)))) 
        + synapse[2]*(1/(x*s_2*math.sqrt(2*math.pi)))*(np.exp(-(math.log(x,math.e)-mu_2)**2/(2*(s_2**2))))
        error_F = error_F + (abs(F - F_))/F
        error_P = error_F + (abs(P - P_))/P
    M_f = M_f + error_F/len(test_x)/100
    M_p = M_p + error_P/len(test_x)/100
print(M_f/5)
print(M_p/5)
plt.plot(train_x, PDF,'g',label='ND')
plt.xlabel('GD,a=1,b=2')
plt.ylabel('PDF')