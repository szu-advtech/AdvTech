from .DataLoader import DataLoader
import torch.optim as optim
import torch
import csv
from universal.algos.bah import BAH
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import heapq
import datetime
import os

class PortfolioRisk_weight():
    def __init__(self, b_expectedReturn, col, batchsize=30):
        '''

        :param b_expectedReturn: is the expected return, it is a real number.
        :param trainloader:  Pytorch trainloader
        :param testloader:  Pytorch testLoader
        :param col:  the dimension of the price ratio data frame
        :param savefile: the file to save the results of each step's training
        '''
        self.b_expectedReturn = b_expectedReturn


        self.IterTimeLossList = None
        self.testLossList = None


        # the file to save training result:
        self.savefile = os.getcwd() + '/resultSave/' + str(datetime.datetime.now()) + '.csv'

        self.file = open(self.savefile, 'w')
        self.csv_writer = csv.writer(self.file)

        self.batchsize = batchsize
        self.col = col
        # random weight
        self.b = torch.rand(col, 1).double()  # normal distribution
        # self.b = torch.ones(self.col, 1).double() / self.col
        self.b.requires_grad = True
        self.optimizer = optim.SGD([self.b], lr=1e-06, momentum=0.9)

    def _getLoader(self, dfHistory, numOfStocks):  # createTrainLoader
        self.topStocks = numOfStocks
        # self.col = dfHistory.shape[1]

        history_numpy = np.array(dfHistory)
        history_tensor = torch.from_numpy(history_numpy)
        loader = torch.utils.data.DataLoader(history_tensor, batch_size=self.batchsize,
                                             shuffle=True, num_workers=2)
        self.trainloader = loader

        # history_batch_numpy = np.array(dfHistory[-self.batchsize:])
        # history_batch_tensor = torch.from_numpy(history_batch_numpy)
        # loader_batch = torch.utils.data.DataLoader(history_batch_tensor, batch_size=self.batchsize,
        #                                            shuffle=True, num_workers=2)
        #
        # self.train_batchloader = loader_batch

    def getBalance(self, dfHistory, numOfStocks):
        '''
        :param dfHistory: dataframe for the history data
        :return: balance whose elements are all positive and the sum is 1
        '''
        self._getLoader(dfHistory, numOfStocks)

        balance = self._trainData()
        balance_numpy = balance.detach().numpy()
        balance_list = []
        for i in balance_numpy:
            balance_list.append(i[0])
        high_balance_index = list(map(balance_list.index, heapq.nlargest(self.topStocks, balance_list)))
        low_balance_index = list(map(balance_list.index, heapq.nsmallest(self.topStocks, balance_list)))


        # balance_list = [0] * self.col
        # for index in high_balance_index:
        #     balance_list[index] = 0.8
        # for index in low_balance_index:
        #     balance_list[index] = 0.2


        balance_series = pd.Series(balance_list, index=dfHistory.columns)

        return balance_series



    def getTopStocks(self, dfHistory, numOfStocks):
        '''
        :param dfHistory: dataframe for the history data
        :return: balance whose elements are all positive and the sum is 1
        '''
        # self.nTopStocks = nTopStocks
        self._getLoader(dfHistory, numOfStocks)  # _createTrainLoader(dfHistory)

        balance = self._trainData_topStocks()
        balance = torch.softmax(balance, dim=0)
        # balance_numpy = balance.detach().numpy()
        balance_list = []
        for i in balance:
            balance_list.append(i[0])
        balance_series = pd.Series(balance_list, index=dfHistory.columns)

        return balance_series

    def _trainData(self):

        # for (  e in epochs)
        self.optimizer.zero_grad()
        loss = self._loss(self.b, self.trainloader).cuda()
        # loss = self._loss_batch(self.b, self.train_batchloader).cuda()
        loss.backward()
        self.optimizer.step()
        self.csv_writer.writerow(["day", "index of top", "weights of top"])

        # Normalized and calculate top stocks' index and value
        b_normal = torch.softmax(self.b, dim=0)
        b_list = list(b_normal)
        top_index = list(map(b_list.index, heapq.nlargest(len(self.b), b_list)))
        top_value = torch.tensor(heapq.nlargest(len(self.b), b_list))


        # last_index = list(map(b_list.index, heapq.nsmallest(self.topStocks, b_list)))
        # last_value = torch.tensor(heapq.nsmallest(self.topStocks, b_list))

        self.csv_writer.writerow(
            [
                # build a list of
                [str(0) + '--' + str(len(self.trainloader) * self.batchsize)],
                top_index,
                list(top_value.detach().numpy())
            ])

        y = torch.softmax(self.b, dim=0)

        # print('y:', y)
        return self.b


    def Multithreadingtrain(self, numOfThreads, numOfStocks):
        '''
        numOfThreads : the number of threads.
        numOfStocks: the number of selected stocks.
        '''


        topkList = []
        for i in range(numOfThreads):
            print("numOfThreads = ", i)

            x = self._trainData()
            y = torch.softmax(x, dim=0)

            a = list(y)
            re1 = map(a.index, heapq.nlargest(numOfStocks, a))
            re2 = torch.tensor(heapq.nlargest(numOfStocks, a))
            topkList.append((list(re1), list(re2.detach().numpy())))

        return topkList



    def _trainData_topStocks(self):

        self.optimizer.zero_grad()
        loss = self._loss(self.b, self.trainloader).cuda()
        # yjf. self.b  put into the GPU
        # loss = self._loss_batch(self.b, self.train_batchloader).cuda()
        loss.backward()
        self.optimizer.step()

        # Normalized and calculate top stocks' index and value
        self.csv_writer.writerow(["day", "index of top", "index of low", "weights of top", "weights of low"])
        b_normal = torch.softmax(self.b, dim=0)
        b_list = list(b_normal)
        top_index = list(map(b_list.index, heapq.nlargest(self.topStocks, b_list)))
        last_index = list(map(b_list.index, heapq.nsmallest(self.topStocks, b_list)))
        top_value = torch.tensor(heapq.nlargest(self.topStocks, b_list))
        last_value = torch.tensor(heapq.nsmallest(self.topStocks, b_list))
        self.csv_writer.writerow(
            [
                # build a list of
                [str(0) + '--' + str(len(self.trainloader) * self.batchsize)],
                 top_index, last_index,
                 list(top_value.detach().numpy()),
                list(last_value.detach().numpy())
            ])



        y = np.zeros((len(self.b), 1))
        for i in top_index:  # like [2, 5]
            # y[i] = 1 / (self.topStocks * 2)
            y[i] = 0.8 / len(top_index)
        for j in last_index:
            # y[j] = 1 / (self.topStocks * 2)
            y[j] = 0.2 / len(last_index)

        return y



    def _loss(self, x, loader):
        '''
            calculate the average loss between 0 and current batch
        :param x:  weight, self.b

        :param loader:  loader for the dataset from which we compute the loss.
        :return:  loss a number
        '''


        y = torch.softmax(x, dim=0)
        dotProds = torch.tensor(0.0)
        reguItem = 0.0
        rList = list(torch.arange(0, 1, 0.01))
        rList.sort()
        alpha_r = 0.0
        for j, data in enumerate(loader):
            ksaiBatch = data  # get one batch
            Ex = torch.matmul(ksaiBatch, y)     # ksaiBatch?
            # Ex:  (batchSize, 1)
            dotProds = dotProds + torch.sum(Ex)  # sum over each samples in each batch

            for r in rList:
                # reguItem += (1 / self.batchSize * torch.sum(torch.exp(b - Ex - r)) - alpha_r)
                reguItem += torch.sum(torch.exp(self.b_expectedReturn - Ex - r))
                # torch.sum over batchSize rows of each batch


        for r in rList:
            alpha_r += 1.0 / ((r + 1.1) ** 4)

        # sum1 = dotProds / ((ibatch + 1) * self.batchsize)
        datasize = len(loader) * self.batchsize
        sum1 = dotProds/datasize

        sum2 = reguItem / datasize
        sum3 = 1e+08 * (sum2 - alpha_r)

        loss = -sum1 + sum3
        return loss



    def _loss_batch(self, x, loader):
        '''
            calculate the average loss between 0 and current batch
        :param x:

        :param loader:  loader for the dataset from which we compute the loss.
        :return:
        '''

        y = torch.softmax(x, dim=0)
        dotProds = torch.tensor(0.0)
        reguItem = 0.0
        rList = list(torch.arange(0, 1, 0.1))
        rList.sort()
        alpha_r = 0.0
        for j, data in enumerate(loader):
            ksaiBatch = data
            Ex = torch.matmul(ksaiBatch, y)
            dotProds = dotProds + torch.sum(Ex)


            for r in rList:
                # reguItem += (1 / self.batchSize * torch.sum(torch.exp(b - Ex - r)) - alpha_r)
                reguItem += torch.sum(torch.exp(self.b_expectedReturn - Ex - r))


        for r in rList:
            alpha_r += 1.0 / ((r + 1.1) ** 4)

        sum1 = dotProds / (self.batchsize)

        sum2 = reguItem / (self.batchsize)

        sum3 = 1e+08 * (sum2 - alpha_r)

        loss = -sum1 + sum3


        return loss

    def getData(self, dfHistory, numOfStocks):
        """
        dfHistory : dataframe ,dataset
        numOfStocks:  select number of stocks
        retrun: a dataframe , choose top stocks columns.
        """

        self._getLoader(dfHistory, numOfStocks)

        balance = self._trainData()
        balance_numpy = balance.detach().numpy()
        balance_list = []
        for i in balance_numpy:
            balance_list.append(i[0])
        high_balance_index = list(map(balance_list.index, heapq.nlargest(self.topStocks, balance_list)))
        low_balance_index = list(map(balance_list.index, heapq.nsmallest(self.topStocks, balance_list)))

        index = high_balance_index + low_balance_index
        dfData = dfHistory.iloc[:, index]

        self.dfData = dfData
        return dfData



















    @staticmethod
    def test_train():

        t = PortfolioRisk_weight(0.9)
        print((os.path.dirname(os.getcwd())))
        path = os.path.dirname(os.getcwd()) + '/data/djia.pkl'
        df = pd.read_pickle(path)
        print(t.getBalance(df))



if __name__ == '__main__':
    PortfolioRisk_weight.test_train()


