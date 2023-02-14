# 使用torch.nn包来构建神经网络.
from pathlib import Path
import random
from time import time
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import logging
import torch
import pandas as pd
from numpy import sqrt
from torch import optim
from torch.autograd._functions import tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from sklearn.metrics import classification_report, mean_absolute_error, mean_squared_error
import stats as sts
from torchvision.transforms import transforms
from sklearn.preprocessing import RobustScaler


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MyDataset(Dataset):
    def __init__(self, csv_file,score, transform=None):
        self.data = np.load(csv_file,allow_pickle=True)
        self.transform = transform
        self.score=score

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        m_data = self.data[idx]
        label=float(m_data[2]/self.score)
        feature=np.array(m_data[0])
        person=m_data[1]
        if self.transform:
            label = self.transform(label)
            feature=self.transform(feature)

        return feature,label,person

class MLP(nn.Module):
    def __init__(self,feature_num):
        super().__init__()
        self.model=nn.Sequential(
            nn.Linear(feature_num,32),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(32,32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        x=self.model(x)
        return x


class CustomLoss(nn.Module):
    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self, x, y):
        mae=nn.L1Loss()(x,y)
        return mae


def Concordance_cc(x,y):
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc

def Person_cc(x,y):
    sxy = np.sum((x - x.mean()) * (y - y.mean())) / x.shape[0]
    rho = sxy / (np.std(x) * np.std(y))
    return rho

def training_AVEC2014(train_file,test_file,feature_num,device_num,epoches,score):
    seed_torch()
    torch.cuda.set_device(device_num)
    device = torch.device("cuda")
    epoches=epoches
    feature_num=feature_num
    weight_decay=1e-3
    begin=time()

    train_file=train_file
    test_file=test_file

    train_set=MyDataset(train_file,score)
    test_set=MyDataset(test_file,score)

    train_batchsize=16
    test_batchsize=10000

    train=DataLoader(train_set, batch_size=3000, shuffle=False, num_workers=1)
    train_tj,tj_label,_=next(iter(train))

    tmean=train_tj.mean(axis=0)
    tstd=train_tj.std(axis=0)
    #
    #
    trainloader = DataLoader(train_set, batch_size=train_batchsize, shuffle=True, num_workers=1)
    testloader = DataLoader(test_set, batch_size=test_batchsize, shuffle=False, num_workers=1)
    train_len=len(trainloader.dataset)
    print(train_len)
    test_len=len(testloader.dataset)

    test_data_iter = iter(testloader)
    test_image, test_label,person = next(test_data_iter)
    test_image=(test_image-tmean)/tstd

    test_image=test_image.to(torch.float32)
    test_label=test_label.to(torch.float32)
    test_image=test_image.to(device)
    test_label=test_label.to(device)


    net = MLP(feature_num).to(device)
    loss_function=CustomLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-3)

    train_best=100
    train_epoch=0
    train_train=np.zeros(4)
    train_test=np.zeros(4)
    train_pre=np.zeros(50)
    test_best=100
    test_epoch=0
    test_train=np.zeros(4)
    test_test=np.zeros(4)
    test_pre=np.zeros(50)

    for epoch in range(epoches):  # 一个epoch即对整个训练集进行一次训练
        net.train()
        running_loss = 0.0
        time_start = time()
        num=0
        train_loss=0
        mae=np.zeros(2)
        rmse=np.zeros(2)
        pcc=np.zeros(2)
        ccc=np.zeros(2)

        mae_t=0
        rmse_t=0
        pcc_t=0
        ccc_t=0

        for step, data in enumerate(trainloader, start=0):
            inputs, labels,_ = data
            inputs=(inputs-tmean)/tstd
            inputs=inputs.to(torch.float32)
            labels=labels.to(torch.float32)
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            output=outputs.reshape(-1,)
            loss = loss_function(output, labels)
            label=labels.cpu().detach().numpy()
            output=output.cpu().detach().numpy()
            label=label*score
            output=output*score
            mae_t+= mean_absolute_error(label, output)
            rmse_t+=sqrt(mean_squared_error(label, output))
            pcc_t+=Person_cc(label, output)
            ccc_t+=Concordance_cc(label,output)
            if weight_decay>0:
                l1=torch.tensor(0.).to(device)
                for param in net.parameters():
                    l1+=torch.sum(torch.abs(param))
                loss+=weight_decay*l1
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num=step+1
        train_loss=running_loss/num
        mae[0]=mae_t/num
        rmse[0]=rmse_t/num
        pcc[0]=pcc_t/num
        ccc[0]=ccc_t/num



        net.eval()
        with torch.no_grad():
            outputs = net(test_image)
            pre=outputs.reshape(-1,)
            label=test_label.cpu().detach().numpy()
            pre=pre.cpu().detach().numpy()
            label=label*score
            pre=pre*score
            mae[1]=mean_absolute_error(label, pre)
            rmse[1]=sqrt(mean_squared_error(label, pre))
            pcc[1]=Person_cc(label, pre)
            ccc[1]=Concordance_cc(label,pre)
        if mae[0] < train_best:
            train_best = mae[0]
            train_epoch=epoch
            train_train[0]=mae[0]
            train_train[1]=rmse[0]
            train_train[2]=pcc[0]
            train_train[3]=ccc[0]
            train_test[0]=mae[1]
            train_test[1]=rmse[1]
            train_test[2]=pcc[1]
            train_test[3]=ccc[1]
            train_pre=pre

        if mae[1]<test_best:
            test_best =mae[1]
            test_epoch = epoch
            test_train[0] = mae[0]
            test_train[1] = rmse[0]
            test_train[2] = pcc[0]
            test_train[3] = ccc[0]
            test_test[0] = mae[1]
            test_test[1] = rmse[1]
            test_test[2] = pcc[1]
            test_test[3] = ccc[1]
            test_pre = pre

        print('[%d] train_loss: %.3f train_mae: %.3f   train_rmse: %.3f  train_pcc: %.3f  train_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
              (epoch + 1, train_loss,mae[0],rmse[0],pcc[0],ccc[0]))
        print('test_mae: %.3f   test_rmse: %.3f  test_pcc: %.3f  test_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
              (mae[1],rmse[1],pcc[1],ccc[1]))
        print('%f s' % (time() - time_start))  # 打印耗时

    print('训练集最好表现 epoch:%d train_mae: %.3f   train_rmse: %.3f  train_pcc: %.3f  train_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (train_epoch,train_train[0],train_train[1],train_train[2],train_train[3]))
    print('相应的测试集情况 test_mae: %.3f   test_rmse: %.3f  test_pcc: %.3f  test_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (train_test[0],train_test[1],train_test[2],train_test[3]))
    print(train_pre.tolist())

    print('测试集最好表现时训练集情况 epoch:%d train_mae: %.3f   train_rmse: %.3f  train_pcc: %.3f  train_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (test_epoch,test_train[0],test_train[1],test_train[2],test_train[3]))
    print('相应的测试集情况 test_mae: %.3f   test_rmse: %.3f  test_pcc: %.3f  test_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (test_test[0],test_test[1],test_test[2],test_test[3]))
    print(test_pre.tolist())
    print(label.tolist())
    print(person)


    print('Finished Training')

    end=time()
    print('%f s' % (end - begin))


def training_AVEC2019(train_file,test_file,dev_file,feature_num,device_num,epoches,score):
    seed_torch()
    torch.cuda.set_device(device_num)
    device = torch.device("cuda")
    epoches=epoches
    feature_num=feature_num
    weight_decay=0
    begin=time()

    train_file=train_file
    test_file=test_file
    dev_file=dev_file

    train_set=MyDataset(train_file,score)
    test_set=MyDataset(test_file,score)
    dev_set=MyDataset(dev_file,score)

    train_batchsize=16
    test_batchsize=1000
    dev_batchsize=1000

    train=DataLoader(train_set, batch_size=3000, shuffle=False, num_workers=1)
    train_tj,tj_label,_=next(iter(train))
    print("训练集大小：",train_tj.shape)

    tmean=train_tj.mean(axis=0)
    tstd=train_tj.std(axis=0)
    #
    #
    trainloader = DataLoader(train_set, batch_size=train_batchsize, shuffle=True, num_workers=1)
    testloader = DataLoader(test_set, batch_size=test_batchsize, shuffle=False, num_workers=1)
    devloader=DataLoader(dev_set, batch_size=dev_batchsize, shuffle=False, num_workers=1)

    test_data_iter = iter(testloader)
    test_image, test_label,_ = next(test_data_iter)
    test_image=(test_image-tmean)/tstd

    dev_data_iter = iter(devloader)
    dev_image, dev_label,_ = next(dev_data_iter)
    dev_image=(dev_image-tmean)/tstd

    test_image=test_image.to(torch.float32)
    test_label=test_label.to(torch.float32)
    test_image=test_image.to(device)
    test_label=test_label.to(device)

    dev_image=dev_image.to(torch.float32)
    dev_label=dev_label.to(torch.float32)
    dev_image=dev_image.to(device)
    dev_label=dev_label.to(device)

    net = MLP(feature_num).to(device)  # 定义训练的网络模型
    loss_function=CustomLoss().to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)  # 定义优化器（训练参数，学习率）

    train_best=1
    train_epoch=0
    train_train=np.zeros(4)
    train_test=np.zeros(4)
    train_dev=np.zeros(4)
    train_pre=np.zeros(56)
    train_pred=np.zeros(56)

    test_best=0
    test_epoch=0
    test_train=np.zeros(4)
    test_test=np.zeros(4)
    test_dev=np.zeros(4)
    test_pre=np.zeros(56)
    test_pred=np.zeros(56)

    dev_best=0
    dev_epoch=0
    dev_train=np.zeros(4)
    dev_test=np.zeros(4)
    dev_dev=np.zeros(4)
    dev_pre=np.zeros(56)
    dev_pred=np.zeros(56)

    for epoch in range(epoches):
        net.train()
        running_loss = 0.0
        time_start = time()
        num=0
        train_loss=0
        mae=np.zeros(3)
        rmse=np.zeros(3)
        pcc=np.zeros(3)
        ccc=np.zeros(3)

        mae_t=0
        rmse_t=0
        pcc_t=0
        ccc_t=0

        for step, data in enumerate(trainloader, start=0):
            inputs, labels,_ = data
            inputs=(inputs-tmean)/tstd
            inputs=inputs.to(torch.float32)
            labels=labels.to(torch.float32)
            inputs=inputs.to(device)
            labels=labels.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            output=outputs.reshape(-1,)
            loss = loss_function(output, labels)
            label=labels.cpu().detach().numpy()
            output=output.cpu().detach().numpy()
            label=label*score
            output=output*score
            mae_t+= mean_absolute_error(label, output)
            rmse_t+=sqrt(mean_squared_error(label, output))
            pcc_t+=Person_cc(label, output)
            ccc_t+=Concordance_cc(label,output)
            if weight_decay>0:
                l1=torch.tensor(0.).to(device)
                for param in net.parameters():
                    l1+=torch.sum(torch.abs(param))
                loss+=weight_decay*l1
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            num=step+1
        train_loss=running_loss/num
        mae[0]=mae_t/num
        rmse[0]=rmse_t/num
        pcc[0]=pcc_t/num
        ccc[0]=ccc_t/num


        net.eval()
        with torch.no_grad():
            outputs = net(test_image)
            pre_test=outputs.reshape(-1,)
            label_test=test_label.cpu().detach().numpy()
            pre_test=pre_test.cpu().detach().numpy()
            label_test=label_test*score
            pre_test=pre_test*score
            mae[1]=mean_absolute_error(label_test, pre_test)
            rmse[1]=sqrt(mean_squared_error(label_test, pre_test))
            pcc[1]=Person_cc(label_test, pre_test)
            ccc[1]=Concordance_cc(label_test,pre_test)

            outputs = net(dev_image)
            pre_dev = outputs.reshape(-1, )
            label_dev=dev_label.cpu().detach().numpy()
            pre_dev=pre_dev.cpu().detach().numpy()
            label_dev=label_dev*score
            pre_dev=pre_dev*score
            mae[2] = mean_absolute_error(label_dev, pre_dev)
            rmse[2] = sqrt(mean_squared_error(label_dev, pre_dev))
            pcc[2] = Person_cc(label_dev, pre_dev)
            ccc[2] = Concordance_cc(label_dev,pre_dev)

        if train_loss < train_best:
            train_best = train_loss
            train_epoch=epoch
            train_train[0]=mae[0]
            train_train[1]=rmse[0]
            train_train[2]=pcc[0]
            train_train[3]=ccc[0]
            train_test[0]=mae[1]
            train_test[1]=rmse[1]
            train_test[2]=pcc[1]
            train_test[3]=ccc[1]
            train_dev[0]=mae[2]
            train_dev[1]=rmse[2]
            train_dev[2]=pcc[2]
            train_dev[3]=ccc[2]

            train_pre=pre_test
            train_pred=pre_dev

        if ccc[1]>test_best:
            test_best = ccc[1]
            test_epoch = epoch
            test_train[0] = mae[0]
            test_train[1] = rmse[0]
            test_train[2] = pcc[0]
            test_train[3] = ccc[0]
            test_test[0] = mae[1]
            test_test[1] = rmse[1]
            test_test[2] = pcc[1]
            test_test[3] = ccc[1]
            test_dev[0]=mae[2]
            test_dev[1]=rmse[2]
            test_dev[2]=pcc[2]
            test_dev[3]=ccc[2]
            test_pre = pre_test
            test_pred=pre_dev

        if ccc[2]>dev_best:
            dev_best = ccc[2]
            dev_epoch = epoch
            dev_train[0] = mae[0]
            dev_train[1] = rmse[0]
            dev_train[2] = pcc[0]
            dev_train[3] = ccc[0]
            dev_test[0] = mae[1]
            dev_test[1] = rmse[1]
            dev_test[2] = pcc[1]
            dev_test[3] = ccc[1]
            dev_dev[0]=mae[2]
            dev_dev[1]=rmse[2]
            dev_dev[2]=pcc[2]
            dev_dev[3]=ccc[2]
            dev_pre = pre_test
            dev_pred=pre_dev





        print('[%d] train_loss: %.3f train_mae: %.3f   train_rmse: %.3f  train_pcc: %.3f  train_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
              (epoch + 1, train_loss,mae[0],rmse[0],pcc[0],ccc[0]))
        print('test_mae: %.3f   test_rmse: %.3f  test_pcc: %.3f  test_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
              (mae[1],rmse[1],pcc[1],ccc[1]))
        print('dev_mae: %.3f   dev_rmse: %.3f  dev_pcc: %.3f  dev_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
              (mae[2],rmse[2],pcc[2],ccc[2]))
        print('%f s' % (time() - time_start))  # 打印耗时

    print('训练集最好表现 epoch:%d train_mae: %.3f   train_rmse: %.3f  train_pcc: %.3f  train_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (train_epoch,train_train[0],train_train[1],train_train[2],train_train[3]))
    print('相应的测试集情况 test_mae: %.3f   test_rmse: %.3f  test_pcc: %.3f  test_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (train_test[0],train_test[1],train_test[2],train_test[3]))
    print('相应的验证集情况 dev_mae: %.3f   dev_rmse: %.3f  dev_pcc: %.3f  dev_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (train_dev[0],train_dev[1],train_dev[2],train_dev[3]))
    print("测试集预测情况：",train_pre.tolist())
    print("验证集预测情况：",train_pred.tolist())

    print('测试集最好表现时训练集情况 epoch:%d train_mae: %.3f   train_rmse: %.3f  train_pcc: %.3f  train_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (test_epoch,test_train[0],test_train[1],test_train[2],test_train[3]))
    print('相应的测试集情况 test_mae: %.3f   test_rmse: %.3f  test_pcc: %.3f  test_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (test_test[0],test_test[1],test_test[2],test_test[3]))
    print('相应的验证集情况 dev_mae: %.3f   dev_rmse: %.3f  dev_pcc: %.3f  dev_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (test_dev[0],test_dev[1],test_dev[2],test_dev[3]))
    print("测试集预测情况：",test_pre.tolist())
    print("验证集预测情况：",test_pred.tolist())

    print('验证集最好表现时训练集情况 epoch:%d train_mae: %.3f   train_rmse: %.3f  train_pcc: %.3f  train_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (dev_epoch,dev_train[0],dev_train[1],dev_train[2],dev_train[3]))
    print('相应的测试集情况 test_mae: %.3f   test_rmse: %.3f  test_pcc: %.3f  test_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (dev_test[0],dev_test[1],dev_test[2],dev_test[3]))
    print('相应的验证集情况 dev_mae: %.3f   dev_rmse: %.3f  dev_pcc: %.3f  dev_ccc: %.3f'%  # 打印epoch，step，loss，accuracy
          (dev_dev[0],dev_dev[1],dev_dev[2],dev_dev[3]))
    print("测试集预测情况：",dev_pre.tolist())
    print("验证集预测情况：",dev_pred.tolist())

    print("测试集的真实值：",label_test.tolist())
    print("验证集的真实值：",label_dev.tolist())


    print('Finished Training')

    end=time()
    print('%f s' % (end - begin))

if __name__ == "__main__":
    #training_AVEC2014( "AVEC2014/combine/features_delete/Training/Northwind_cfs50.npy", "AVEC2014/combine/features_delete/Testing/Northwind_cfs50.npy",50,4,100,64)
    training_AVEC2019( "AVEC2019/combine/features_deletetrain_cfs47.npy", "AVEC2019/combine/features_deletetest_cfs47.npy","AVEC2019/combine/features_deletedev_cfs47.npy",47,4,100,25)