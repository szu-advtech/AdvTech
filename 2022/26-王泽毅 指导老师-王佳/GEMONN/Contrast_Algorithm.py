import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import numpy as np

class Net(nn.Module):
    def __init__(self, InputDim, HiddenNum, OutputDim):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(InputDim, HiddenNum)
        self.fc2 = nn.Linear(HiddenNum, OutputDim, bias=False)

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
                               transform=transforms.ToTensor(),  # 转换成tensor变量
                               download=False)  # 不从网络上download图片
    T_Dim = np.array(train_dataset.train_data.shape)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch,
                                               shuffle=True)  # 将数据打乱
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch,
                                              shuffle=True)  # 将数据打乱
    Dim = T_Dim[1] * T_Dim[2]
    return Dim, train_loader, test_loader

def train_layer_wise_autoencoder(Dim, HiddenNum, trainloader, Model_trans, Gene):
    import time
    since = time.time()
    Model = Net(Dim, HiddenNum, Dim).cuda()
    count = 0
    Flag = False
    Loss = nn.MSELoss()
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.001)
    save_loss = []
    for epoch in range(Gene):
        running_loss = 0.0
        if Flag == True:
            break
        for i, data in enumerate(trainloader, 0):
            if count >= 25000:
                Flag = True
                break
            inputs, labels = data
            inputs = inputs.view(inputs.size(0), -1).cuda()
            if Model_trans is not None:
                _, inputs = Model_trans(inputs)
            optimizer.zero_grad()
            outputs, _ = Model(inputs)
            loss = Loss(outputs, inputs)
            save_loss.append(loss.data.cpu().numpy())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            count += 1
            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f, time:%.3f' % (epoch + 1, i + 1, running_loss / 100, time.time() - since))
                running_loss = 0.0
    print('Finished Training')
    return Model.get_weights()

if __name__ == '__main__':
    Dim, trainloader, testloader = LoadData(batch=128)
    weights = []
    Model = None
    layer_1_weights = train_layer_wise_autoencoder(Dim, 500, trainloader, Model, Gene=100)
    weights.append(layer_1_weights.cpu().data)
    path = 'result/Adam0001.txt'
    file = open(path, 'w')
    for i in range(len(weights)):
        t = weights[i]
        t = t.cpu().numpy()
        Nums = [t_i[0] for t_i in t]
        for j in Nums:
            file.write(str(j) + " ")
    file.close()