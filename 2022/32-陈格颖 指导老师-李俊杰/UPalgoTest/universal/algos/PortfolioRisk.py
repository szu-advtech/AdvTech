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
import os
import datetime

class PortfolioRisk():
    def __init__(self, b_expectedReturn, trainloader, testloader, col):
        '''

        :param b_expectedReturn: is the expected return, it is a real number.
        :param trainloader:  Pytorch trainloader
        :param testloader:  Pytorch testLoader
        :param col:  the dimension of the price ratio data frame
        :param savefile: the file to save the results of each step's training
        '''

        self.b = b_expectedReturn
        self.trainloader = trainloader
        self.testloader = testloader
        self.IterTimeLossList = None
        self.testLossList = None
        self.col = col

        # the file to save training result:
        self.savefile = os.getcwd() + '/olmar_old_data/' + str(datetime.datetime.now()) + 'choose stocks data.csv'
        self.file = open(self.savefile, 'w')
        self.csv_writer = csv.writer(self.file)

        self.batchSize = trainloader.batch_size
        self.xAllocationRate = None

    def loss(self, x, i, loader):
        '''
            calculate the average loss between 0 and current batch
        :param x:
        :param i:
        :param loader:
        :return:
        '''

        y = torch.softmax(x, dim=0)
        dotProds = torch.tensor(0.0)
        reguItem = 0.0
        rList = list(torch.arange(0, 1, 0.1))
        rList.sort()
        alpha_r = 0.0
        for j, data in enumerate(loader):
            if j > (i + 1):
                break


            ksaiBatch = data
            Ex = torch.matmul(ksaiBatch, y)
            dotProds = dotProds + torch.sum(Ex)


            for r in rList:
                # reguItem += (1 / self.batchSize * torch.sum(torch.exp(b - Ex - r)) - alpha_r)
                reguItem += torch.sum(torch.exp(self.b - Ex - r))


        for r in rList:
            alpha_r += 1.0 / ((r + 1.1) ** 4)

        sum1 = dotProds / ((i+1) * self.batchSize)

        sum2 = reguItem / ((i+1) * self.batchSize)

        sum3 = 1e+08 * (sum2 - alpha_r)

        loss = -sum1 + sum3

        # dotProd = dotProd.view(-1).item()

        # reguItem = 0.0
        # rList = list(np.random.uniform(0, 1, 20))
        # rList.sort()
        # for r in rList:
        #     alpha_r = 1.0 / ((r + 1.1) ** 4)
        #     reguItem += 1e+08 * (1 / (self.batchSize * (i + 1)) * torch.sum(torch.exp(b - dotProds - r)) - alpha_r)

        # loss = -1 * dotProds + reguItem
        # loss = torch.div(loss, i+1)
        # print('loss: ', loss)
        return loss



    def testLoss(self, x):
        '''
        calculate entire testDataset's average loss
        :param x:
        :return:
        '''

        loss = self.loss(x, len(self.testloader)-1, self.testloader)

        return loss


    def _trainOneThread(self, topNumOfStocks, epochNum=6):

        self.testLossList = []
        self.IterTimeLossList = []
        self.csv_writer.writerow(["epoch", "day", "index of top", "weights", "ARLtop_Sharptop_MDDtop"])
        x = torch.rand(self.col, 1).double()  # normal distribution
        # x = torch.ones(self.col, 1).double()/self.col        # mean distribution
        x.requires_grad = True
        optimizer = optim.SGD([x], lr=1e-09, momentum=0.9)

        for epoch in range(epochNum):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader):
                # ksai = data                  # size of ksai is batchSize(4) * 30(cols)

                # zero the parameter gradients
                optimizer.zero_grad()

                y = torch.softmax(x, dim=0)
                a = list(y)
                re1 = list(map(a.index, heapq.nlargest(topNumOfStocks, a)))
                re2 = torch.tensor(heapq.nlargest(topNumOfStocks, a))

                ARL_Sharp_MDD = self.topkStocks(i, topNumOfStocks)

                self.csv_writer.writerow([epoch, [str(i * self.batchSize) + '--' + str((i + 1) * self.batchSize)], re1,
                                          list(re2.detach().numpy()), ARL_Sharp_MDD])

                # forward + backward + optimize
                loss = self.loss(x, i, self.trainloader).cuda()
                # print('loss:', loss)
                loss.backward()  # backward should be size of loss

                optimizer.step()  # x <- x_old - lr * nabla(loss/x)

                record_test = (epoch * len(self.trainloader) + i, self.testLoss(x))
                self.testLossList.append(record_test)

                running_loss += loss
                if (i + 1) % 3 == 0:
                    record_train = (epoch * len(self.trainloader) + i, running_loss / 5)
                    self.IterTimeLossList.append(record_train)
                    running_loss = 0.0

                # print top max
                # y = x.view(size=(self.col,))
                # y = torch.softmax(x, dim=0)
                # y = y.view(size=(self.col,))
                # xs, _ = torch.sort(y)
                # print('i: ', i, 'maxIds: ', torch.argmax(y, dim=0), 'x top max: ', xs[self.col - 1], xs[self.col - 2], '....#nonzero: ', torch.nonzero(xs).size())

            if epoch % 5 == 3:
                lr = optimizer.param_groups[0]['lr']
                print('epoch', epoch, 'learing rate: ....', lr)
                optimizer.param_groups[0]['lr'] = lr / 2
                # for para in optimizer.param_groups:
                #     para['lr'] = lr/2
                #     print('lr:', lr)
        # print('x', x)
        return x


    def train(self, numOfThreads, topNumOfStocks):
        '''
        train trainDataset, getted new weights every batch, then calculate testLoss
        :param savefile:
        :return:
        '''




        topkList = []
        for i in range(numOfThreads):
            print("numOfThreads = ", i)

            x = self._trainOneThread(topNumOfStocks)
            y = torch.softmax(x, dim=0)

            a = list(y)
            re1 = map(a.index, heapq.nlargest(topNumOfStocks, a))
            re2 = torch.tensor(heapq.nlargest(topNumOfStocks, a))
            topkList.append((list(re1), list(re2.detach().numpy())))

            # y = torch.softmax(x, dim=0)
            # self.xAllocationRate = y

            # self.display_train_test_loss()

        self.file.close()

        # print(y)
        # a = list(y)
        # re1 = map(a.index, heapq.nlargest(3, a))
        # re2 = torch.tensor(heapq.nlargest(3, a))
        #
        # print('index of the top3 :', list(re1))
        # print('value of the top3 :', re2.detach().numpy())

        return topkList

    def topkStocks(self, i, k_stocksnum):

        '''
         calculate ARL,sharp,MDD to evaluate the weights.
        :param maxIndex:
        :return:
        '''

        data = pd.read_pickle(self.originFile)
        index = (i+1) * self.batchSize
        data = data[0:index]
        ARList = []
        SharpeList = []
        MDDList = []
        for k in range(0, self.col):
            one = np.zeros(self.col)
            one[k] = 1
            result_bah = BAH(one).run(data)
            result_bah.fee = 0.0


            ARList.append(result_bah.annualized_return)
            SharpeList.append(result_bah.sharpe)
            MDDList.append(result_bah.max_drawdown)


        # ARList = np.asarray(ARList)
        # SharpeList = np.asarray(SharpeList)
        # MDDList = np.asarray(MDDList)


        ARL_top_index = map(ARList.index, heapq.nlargest(k_stocksnum, ARList))
        ARL_top_value = heapq.nlargest(k_stocksnum, ARList)


        Sharp_top_index = map(SharpeList.index, heapq.nlargest(k_stocksnum, SharpeList))
        Sharp_top_value = heapq.nlargest(k_stocksnum, SharpeList)

        MDD_top_index = map(MDDList.index, heapq.nsmallest(k_stocksnum, MDDList))
        MDD_top_value = heapq.nsmallest(k_stocksnum, MDDList)

        ARL_Sharp_MDD_List = []
        ARL_Sharp_MDD_List.append([list(ARL_top_index), ARL_top_value])
        ARL_Sharp_MDD_List.append([list(Sharp_top_index), Sharp_top_value])
        ARL_Sharp_MDD_List.append([list(MDD_top_index), MDD_top_value])

        return ARL_Sharp_MDD_List

    def getPercentData(self, filename, percent):
        '''
        get percentage of data
        :param filename:
        :param percent:
        :return:
        '''


        df = pd.read_pickle(filename)
        len_of_data = len(df)
        topPer = int(len_of_data * percent)
        df = df[0:topPer]
        ndarray = np.array(df)
        datatens = torch.from_numpy(ndarray)

        return datatens


    def calReturn(self):
        profit = 0.0
        for i, data in enumerate(self.testloader):
            res = torch.matmul(data, self.xAllocationRate)
            profit = profit + torch.sum(res)

        print('profit:', profit / (len(self.testloader) * self.batchSize))

    def vadilateResults(self, filename):

        '''
        vadilate b-ksai * x < r  probability and alpha_r
        :param filename:
        :return:
        '''


        pers = [0.2, 0.4, 0.6, 0.8, 1]


        for per in pers:

            data = self.getPercentData(filename, per)
            res = torch.matmul(data, self.xAllocationRate)   # like 507 * 1
            # rList = list(np.arange(0, 1, 0.1))
            rList = list(np.arange(0, 1, 0.05))
            rList.sort()
            len_of_data = len(res)
            alpha_r = []
            prob = []

            for r in rList:
                sum_ = 0.0
                for i in range(len_of_data):
                    if self.b - res[i] >= r:
                        sum_ = sum_ + 1
                alpha_r.append(1.0 / ((r + 1.1) ** 4))
                prob.append(sum_)

            plt.plot(rList, alpha_r, c='red')
            plt.scatter(rList, prob, label=str(per * 100) + 'per')
            plt.legend()
        # plt.legend(['alpha_r', 'Prob'])
        plt.xlabel('r')
        # plt.ylabel('')
        plt.show()




    def display_train_test_loss(self):
        '''
        display the graph of testdataset and traindataset

        :return:
        '''

        self.IterTimeLossList = np.asarray(self.IterTimeLossList)
        plt.plot(self.IterTimeLossList[:, 0], self.IterTimeLossList[:, 1], c='blue')

        plt.xlabel('IterTimes')
        plt.ylabel('loss')
        plt.legend(['trainLoss'])
        plt.show()
        self.testLossList = np.array(self.testLossList)
        plt.plot(self.testLossList[:, 0], self.testLossList[:, 1], c='red')
        # plt.xlim(-5, 100)
        # plt.ylim(-100, 100)
        plt.xlabel('IterTimes')
        plt.ylabel('loss')
        plt.legend(['testLoss'])
        plt.show()

    def visualOriginalData(self, filename):
        '''
        display the origin data


        :param filename:
        :return:
        '''

        df = pd.read_pickle(filename)
        datanp = np.array(df)
        self.originFile = filename

        row = df.shape[0]
        x = np.arange(row)
        col = df.shape[1]

        cols = 2
        numRows = col // cols

        while numRows >= 10:
            cols += 1
            numRows = np.ceil(col / (cols * 5))

        # start = int(numRows * 100 + cols * 10 + 1)  # 321
        numRows = int(numRows)

        numSubPlots = int(np.ceil(col / 5))

        plt.figure(figsize=(10, 10))
        if col % 5 == 0:
            for time in range(0, numSubPlots):
                plt.subplot(numRows, cols, time+1)
                for i in range(0, 5):
                    plt.plot(x, datanp[:, time * 5 + i], label=time * 5 + i)

                plt.legend()

            plt.show()

        else:

            for time in range(0, numSubPlots-1):
                plt.subplot(numRows, cols, time+1)
                for i in range(0, 5):
                    plt.plot(x, datanp[:, time * 5 + i], label=time * 5 + i)
                plt.legend()

            plt.subplot(numRows, cols, numSubPlots)
            remainder = col % 5

            for j in range(remainder):
                plt.plot(x, datanp[:, (time + 1) * 5 + j], label=(time+1) * 5 + j)

            plt.legend()

            plt.show()


    @staticmethod
    def test_train():

        loader = DataLoader(32, True)
        trainloader,col = loader.getLoaderFromPickle('./Data/djia.pkl')
        loader = DataLoader(32, True)
        testloader, col = loader.getLoaderFromPickle('./Data/djia.pkl')
        t = PortfolioRisk(0.9, trainloader, testloader, col)
        t.visualOriginalData('./Data/djia.pkl')
        t.train('./Result/djia_test.txt')
        t.display_train_test_loss()
        t.vadilateResults('./Data/djia.pkl')



if __name__ == '__main__':
    PortfolioRisk.test_train()

