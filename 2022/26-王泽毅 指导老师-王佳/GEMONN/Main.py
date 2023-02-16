from torch.utils.data import Dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import Evoluationary_algorithm as EA
import argparse, os
import matplotlib.pyplot as plt
import cupy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, InputDimension, HiddenNum, OutputDimension):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(InputDimension, HiddenNum)
        self.fc2 = nn.Linear(HiddenNum, OutputDimension, bias=False)

    def forward(self, X):
        X = torch.sigmoid(self.fc1(X))
        X1 = X
        self.fc2.weight.data = self.fc1.weight.data.t()
        X = torch.sigmoid(self.fc2(X))
        return X, X1

    def Initialization(self, weights_P):
        weights = weights_P[:, :-1]
        bias = weights_P[:, -1]
        self.fc1.weight.data = weights
        self.fc1.bias.data = bias
        self.fc2.weight.data = weights.t()

    def get_weights(self):
        weights_bias = torch.cat((self.fc1.weight.data.t(), self.fc1.bias.data.reshape(1, -1)))
        return weights_bias

def LoadData(batch=64):
    # MNIST dataset
    train_dataset = dsets.MNIST(root='Data/',  # 选择数据的根目录
                                train=True,  # 选择训练集
                                transform=transforms.ToTensor(),  # 转换成tensor变量
                                download=False)  # 不从网络上download图片
    test_dataset = dsets.MNIST(root='Data/',  # 选择数据的根目录
                               train=False,  # 选择训练集
                               transform=transforms.ToTensor(),  # 转换成tensor变量zz
                               download=True)  # 不从网络上download图片
    T_Dimension = cp.array(train_dataset.train_data.shape)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch,
                                               shuffle=True)  # 将数据打乱
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch,
                                              shuffle=False)  # 将数据打乱
    Dimension = T_Dimension[1] * T_Dimension[2]
    return int(Dimension), train_loader, test_loader

def create_dir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except Exception:
            os.makedirs(path)
        print('Make Dir: {}'.format(path))

def Starts(Generations, PopulationSize, HiddenNum, Algorithm, args):
    Dimension, train_loader, test_loader = LoadData(batch=128)  # 导入数据
    Model = Net(Dimension, HiddenNum, Dimension).cuda()  # 生成网络
    Population, Boundary, Coding = EA.Initialization_Population(PopulationSize, Dimension, HiddenNum)  # 生成初始种群
    FunctionValue, Weight_Grad = EA.Evaluation(Population, Dimension, HiddenNum, train_loader, Model)  # 计算个体的稀疏性和梯度
    FrontValue = EA.Non_Dominant_Sorting(FunctionValue, PopulationSize)[0]  # 非主导排序
    CrowdDistance = EA.Crowded_Distance(FunctionValue, FrontValue)  # 计算拥挤距离
    plt.ion()
    for Gene in range(Generations):  # 开始进化迭代
        MatingPool, index = EA.Mating(Population, FrontValue, CrowdDistance)  # 生成交配池
        Weight_Grad_Mating = Weight_Grad[index, :]  # 选择个体的梯度
        Offspring = EA.gSBX(MatingPool, Weight_Grad_Mating, Boundary, Coding, PopulationSize)  # 生成子代个体
        FunctionValue_Offspring, Weight_Grad_Offspring = EA.Evaluation(Offspring, Dimension, HiddenNum, train_loader, Model)  # 计算子代的稀疏性和梯度
        Population = cp.vstack((Population, Offspring))  # 合并亲本和子代
        FunctionValue = np.vstack((FunctionValue, FunctionValue_Offspring))  # 合并亲本和子代稀疏性
        Weight_Grad_Temp = cp.vstack((Weight_Grad, Weight_Grad_Offspring))  # 合并亲本和子代梯度
        Population, FunctionValue, Weight_Grad, FrontValue, CrowdDistance = EA.Environment_Select(
            Population, Weight_Grad_Temp, FunctionValue, PopulationSize)  # 进行环境选择
        if args.plot:  # 画图
            plt.clf()
            plt.plot(FunctionValue[:, 0], FunctionValue[:, 1], "*")
            plt.pause(0.001)
            if (Gene + 1) % 50 == 0:
                plt.savefig(args.save_dir + '/plot_' + str(Gene + 1) + '.png')
        print('\r', Algorithm, "Run:", Gene, "代，minimal loss:" , cp.sort(FunctionValue[:, 1])[:1], end='')
    FunctionValueNon = FunctionValue[(FrontValue == 1)[0], :]
    PopulationNon = Population[(FrontValue == 1)[0], :]
    if args.save:
        np.savetxt(args.save_dir + '/MSE_loss.txt', FunctionValueNon, delimiter=' ')
        np.savetxt(args.save_dir + '/Population.txt', cp.asnumpy(PopulationNon), delimiter=' ')
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GEMONN setting')
    parser.add_argument('--Generations', type=int, default=500, help='The maximal iteration of the algorithm')
    parser.add_argument('--PopulationSize', type=int, default=50, help='The population size')
    parser.add_argument('--HiddenNum', type=int, default=500, help='The number of hidden units of an auto-encoder')
    parser.add_argument('--Algorithm', type=str, default="NSGA-II", help='The used framwork of Evoluitonary Algorithms')
    parser.add_argument('--plot', action='store_true', default=True, help='Plot the function value each generation')
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--save_dir', type=str, default='./result')
    args = parser.parse_args()
    create_dir(args.save_dir)
    Starts(args.Generations, args.PopulationSize, args.HiddenNum, args.Algorithm, args)