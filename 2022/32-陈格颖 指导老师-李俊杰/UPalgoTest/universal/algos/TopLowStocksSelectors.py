import os
import csv
import datetime
import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import heapq
class TopLowStocksSelectors:
    def __init__(self, b_expectedReturn, dataset_nStocks, nTopStocks, nLowStocks, originData, batchsize, loopTrainEpochs=64):
        """

        :param b_expectedReturn:  is the expected return, it is a real number.
        :param dataset_nStocks:  total number of  stocks, is also the dimension of price data.
        :param nTopStocks: the number of top weight
        :param nLowStocks: the number of low weight
        :param loopTrainEpochs:   the number of scans of the datasets.
        :param batchsize:  size of a batch
        """
        self.b_expectedReturn = b_expectedReturn
        self.dataset_nStocks = dataset_nStocks
        self.nTopStocks = nTopStocks
        self.nLowStocks = nLowStocks
        self.loopTrainEpochs = loopTrainEpochs
        self.batchsize = batchsize
        self.originData = originData

        # the file to save training result:
        self.savefile = os.getcwd() + '/resultSave/' + str(datetime.datetime.now()) + '.csv'
        self.file = open(self.savefile, 'w')
        self.csv_writer = csv.writer(self.file)

        # the file to save final result
        self.resfile = os.getcwd() + '/buySave/' + str(datetime.datetime.now()) + '.csv'
        self.refile = open(self.resfile, 'w')
        self.csv_writer2 = csv.writer(self.refile)

        # random weight
        self.b = torch.rand(self.dataset_nStocks, 1).double()
        # self.b = torch.ones(self.dataset_nStocks, 1).double() / self.dataset_nStocks
        self.b.requires_grad = True
        self.optimizer = optim.SGD([self.b], lr=1e-08, momentum=0.9)

        self.rList = list(torch.arange(0, 1, 0.01))
        self.rList.sort()
        self.alpha_r = 0.0
        for r in self.rList:
            self.alpha_r += 1.0 / ((r + 1.1) ** 4)

        self.lookBackBatchSize = 0
        self.lastExpectStockId = float('inf')


    def getTopLowStocks(self, dfHistory):
        """

        :param dfHistory: the price data.
        :return:  return type SERIES.[topStocks_weights,  lowStocks_weights], whose sum is 1.
        """

        balance = self._trainData(dfHistory)
        balance_list = []
        for i in balance:
            balance_list.append(i[0])

        self.csv_writer2.writerow(
            ["stockId", "weight"])

        stockId = np.argmax(balance_list)
        weight = np.max(balance_list)

        self.csv_writer2.writerow([stockId,
                                   weight])

        balance_series = pd.Series(balance_list, index=dfHistory.columns)

        return balance_series

    def _trainData(self, dfHistory):
        """

        :param dfHistory:  price data of all of stocks.
        :return: return weight of a stock.
        """

        self.totalDfHistory = dfHistory
        # getting training, self.lookBackBatchSize == -1, getting data expect stock A, else getting total data,else,.
        if self.lookBackBatchSize == -1:
            self.dfHistory = self.getTrainDataDelExpectStockId(self.lastExpectStockId, dfHistory)
        else:
            self.dfHistory = dfHistory
        self._createTrainLoader(self.dfHistory)

        # get top one stock' weight and index
        # x = self.getTopOneByReuseNN()
        x = self.getTopOneByNewNN()

        Xindex = np.argmax(x)

        # update weight
        if self.lookBackBatchSize == -1:
            # whether stock last X's price down > 0.01
            b = self.isPriceDown(self.batchsize, self.lastExpectStockId)
            if b == True:
                # stock last X' price is down >0.01, update weight
                self.csv_writer.writerow(["stock"])
                self.csv_writer.writerow([np.argmax(x)])
            else:
                # stock last X's price is not down > 0.1, gaining last X current price
                last_Xproduct = self.getRawPrice(self.lastExpectStockId)
                last_XcurrentPrice = last_Xproduct[-1]
                # gaining the new X' current price
                Xproduct = self.getRawPrice(Xindex)
                XcurrentPrice = Xproduct[-1]
                self.csv_writer.writerow(["stock"])
                if last_XcurrentPrice > XcurrentPrice:
                    x = self.x
                    Xindex = self.lastExpectStockId
                    self.csv_writer.writerow([np.argmax(x)])
                else:
                    self.csv_writer.writerow([np.argmax(x)])



        self.lastExpectStockId = Xindex
        self.x = x
        self.lookBackBatchSize = -1

        return x



    def getTrainDataSetExpectStockId(self, stockId, dfHistory):
        """

        :param stockId:
        :param dfHistory:
        :return:
        """
        dataNumpy = np.asarray(dfHistory)
        dataNumpy[:, stockId] = 0
        dataFrame = pd.DataFrame(dataNumpy, columns=self.dfHistory.columns)
        return dataFrame




    def getTrainDataDelExpectStockId(self, stockId, dfHistory):
        """

        :param stockId:
        :param dfHistory:
        :return:
        """


        dataNumpy = np.asarray(dfHistory)
        cols = self.dfHistory.shape[1]
        for i in range(stockId, cols - 1):
            dataNumpy[:, i] = dataNumpy[:, i + 1]
        dataNumpy = np.delete(dataNumpy, -1, axis=1)
        dataFrame = pd.DataFrame(dataNumpy)
        return dataFrame


    def isPriceDown(self, selectedDay, stockId, downPercent=0.01):
        """

        :param selectedDay:
        :param stockId:
        :param downPercent:
        :return:
        """

        product = self.getRawPrice(stockId)
        product = product[-selectedDay:]
        currentPrice = product[-1]
        maxPrice = np.max(product)
        downPer = (maxPrice - currentPrice) / maxPrice

        if downPer > downPercent:
            return True
        return False

    def getRawPrice(self, stockId):
        """

        :param stockId:
        :return:
        """

        # data_numpy = np.asarray(self.totalDfHistory)
        # XdataStock = data_numpy[:, stockId]
        # data_stockId = XdataStock + 1
        # firstDayData = np.asarray(self.originData.head(1))
        # mulit = firstDayData[0][stockId]

        # lenStockId = len(data_stockId)
        # for i in range(lenStockId):
        #     mulit = mulit * data_stockId[i]
        #     product.append(mulit)

        data_numpy = np.asarray(self.originData)
        data_stockId = data_numpy[:, stockId]
        row = self.dfHistory.shape[0]
        product = data_stockId[:row]

        return product





    def getTopOneByReuseNN(self):


        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.csv_writer.writerow(
            ["epoch", "day", "index of top", "weights of top", "index of low", "weights of low", "loss"])
        for epoch in range(self.loopTrainEpochs):
            self.optimizer.zero_grad()
            loss = self._loss(self.b, self.trainloader)
            loss.backward()
            self.optimizer.step()

            # Normalized and calculate top stocks' index and value
            b_normal = torch.softmax(self.b, dim=0)
            b_list = list(b_normal)
            top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
            last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
            top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
            last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))

            self.csv_writer.writerow(
                [
                    # build a list of
                    epoch,
                    [str(0) + '--' + str(len(self.trainloader) * self.batchsize)],
                    top_index, list(top_value.detach().numpy()),
                    last_index, list(last_value.detach().numpy()),
                    loss.cpu().detach().numpy()
                ])

        b_normal = torch.softmax(self.b, dim=0)
        b_list = list(b_normal)

        top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
        last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
        top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
        last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))

        data_numpy = np.asarray(self.dfHistory)
        products = []
        for index in top_index:
            product = self.getRawPrice(index)
            products.append((product[-1], index))
        products_sort = sorted(products, reverse=True)

        x = np.zeros((len(self.b), 1))
        Xindex = products_sort[0][1]
        x[Xindex] = 1

        return x

    def getTopOneByNewNN(self):

        if self.lookBackBatchSize == -1:
            self.b = torch.rand(self.dataset_nStocks-1, 1).double()
            # self.b = torch.ones(self.dataset_nStocks-1, 1).double() / (self.dataset_nStocks -1)
            self.b.requires_grad = True

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.csv_writer.writerow(
            ["epoch", "day", "index of top", "weights of top", "index of low", "weights of low", "loss"])
        for epoch in range(self.loopTrainEpochs):
            self.optimizer.zero_grad()
            loss = self._loss(self.b, self.trainloader)
            loss.backward()
            self.optimizer.step()

            # Normalized and calculate top stocks' index and value
            b_normal = torch.softmax(self.b, dim=0)
            b_list = list(b_normal)
            top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
            last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
            top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
            last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))

            self.csv_writer.writerow(
                [
                    # build a list of
                    epoch,
                    [str(0) + '--' + str(len(self.trainloader) * self.batchsize)],
                    top_index, list(top_value.detach().numpy()),
                    last_index, list(last_value.detach().numpy()),
                    loss.cpu().detach().numpy()
                ])

        b_normal = torch.softmax(self.b, dim=0)
        b_list = list(b_normal)

        top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
        last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
        top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
        last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))

        products = []
        for index in top_index:
            if index >= self.lastExpectStockId:
                index = index + 1
            product = self.getRawPrice(index)
            products.append((product[-1], index))
        products_sort = sorted(products, reverse=True)


        Xindex = products_sort[0][1]
        if self.lookBackBatchSize == -1:
            x = np.zeros((len(self.b)+1, 1))
            x[Xindex] = 1.0
            # if Xindex >= self.lastExpectStockId:
            #     x[Xindex+1] = 1.0
            # else:
            #     x[Xindex] = 1.0
        else:
            x = np.zeros((len(self.b), 1))
            x[Xindex] = 1.0

        return x

    # del stock A data
    def getTopOneByReuseNN_delA(self, currentDay, expectedId, loolBackBatchSize = -1):
        """

        :param currentDay:
        :param expectedId: id of stock.
        :param loolBackBatchSize:
        :return:
        """

        dataNumpy = np.asarray(self.dfHistory)
        cols = self.dfHistory.shape[1]
        for i in range(expectedId, cols-1):
            dataNumpy[:, i] = dataNumpy[:, i+1]
        dataNumpy = np.delete(dataNumpy, -1, axis=1)
        dataFrame = pd.DataFrame(dataNumpy)

        for i in range(expectedId, len(self.b)-1):
            self.b[i] = self.b[i+1]
        self.b = np.delete(self.b.detach().numpy(), -1)
        self.b = torch.tensor(self.b, requires_grad=True)

        self._createTrainLoader(dataFrame)


        # self.b = torch.rand(self.dataset_nStocks-1, 1).double()
        # self.b.requires_grad = True

        # self.b[expectedId] = torch.tensor(0.0, requires_grad=True)
        # self.b.requires_grad = True
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.csv_writer.writerow(
            ["epoch", "day", "index of top", "weights of top", "index of low", "weights of low", "loss"])
        for epoch in range(self.loopTrainEpochs):
            self.optimizer.zero_grad()
            loss = self._loss(self.b, self.trainloader)
            loss.backward()
            self.optimizer.step()

            # Normalized and calculate top stocks' index and value
            b_normal = torch.softmax(self.b, dim=0)
            b_list = list(b_normal)
            top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
            last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
            top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
            last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))

            self.csv_writer.writerow(
                [
                    # build a list of
                    epoch,
                    [str(0) + '--' + str(len(self.trainloader) * self.batchsize)],
                    top_index, list(top_value.detach().numpy()),
                    last_index, list(last_value.detach().numpy()),
                    loss.cpu().detach().numpy()
                ])

        b_normal = torch.softmax(self.b, dim=0)
        b_list = list(b_normal)

        top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
        last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
        top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
        last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))



        y = np.zeros((len(self.b)+1, 1))
        if top_index[0] >= expectedId:
            y[top_index[0]+1] = 1.0
        else:
            y[top_index[0]] = 1.0
        b_last = self.b[-1].detach().numpy()
        self.b = np.insert(self.b.detach().numpy(), -1, b_last)
        for i in reversed(range(expectedId, len(self.b)-1)):
            self.b[i+1] = self.b[i]

        self.b = torch.tensor(self.b, requires_grad=True)
        self.b[expectedId] = 0.0

        return y


    # del stock A data
    def getTopOneByNewNN_delA(self, currentDay, expectedId, loolBackBatchSize = -1):
        """

        :param currentDay:
        :param expectedId: id of stock.
        :param loolBackBatchSize:
        :return:
        """

        dataNumpy = np.asarray(self.dfHistory)
        cols = self.dfHistory.shape[1]
        for i in range(expectedId, cols-1):
            dataNumpy[:, i] = dataNumpy[:, i+1]
        dataNumpy = np.delete(dataNumpy, -1, axis=1)
        dataFrame = pd.DataFrame(dataNumpy)



        self._createTrainLoader(dataFrame)


        self.b = torch.rand(self.dataset_nStocks-1, 1).double()
        self.b.requires_grad = True

        # self.b[expectedId] = torch.tensor(0.0, requires_grad=True)
        # self.b.requires_grad = True
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.csv_writer.writerow(
            ["epoch", "day", "index of top", "weights of top", "index of low", "weights of low", "loss"])
        for epoch in range(self.loopTrainEpochs):
            self.optimizer.zero_grad()
            loss = self._loss(self.b, self.trainloader)
            loss.backward()
            self.optimizer.step()

            # Normalized and calculate top stocks' index and value
            b_normal = torch.softmax(self.b, dim=0)
            b_list = list(b_normal)
            top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
            last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
            top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
            last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))

            self.csv_writer.writerow(
                [
                    # build a list of
                    epoch,
                    [str(0) + '--' + str(len(self.trainloader) * self.batchsize)],
                    top_index, list(top_value.detach().numpy()),
                    last_index, list(last_value.detach().numpy()),
                    loss.cpu().detach().numpy()
                ])

        b_normal = torch.softmax(self.b, dim=0)
        b_list = list(b_normal)

        top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
        last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
        top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
        last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))



        y = np.zeros((len(self.b)+1, 1))
        if top_index[0] >= expectedId:
            y[top_index[0]+1] = 1.0
        else:
            y[top_index[0]] = 1.0

        return y


    def getTopOneByNewNNChoose3(self, currentDay, expectedId, loolBackBatchSize = -1):
        """

        :param currentDay:
        :param expectedId: id of stock.
        :param loolBackBatchSize:
        :return:
        """

        dataNumpy = np.asarray(self.dfHistory)
        # --------------------------------------------
        # dataNumpy[:, expectedId] = float('-inf')
        # dataNumpy[:, expectedId] = 0.0
        dataNumpy[:, expectedId] = -650

        # dataExpectId = dataNumpy[:, expectedId]
        # rows = self.dfHistory.shape[0]
        # ran = - 500
        # gap = ran / rows
        # x = np.arange(0, ran, gap)
        # for i in range(rows):
        #     dataExpectId[i] = x[i]
        dataFrame = pd.DataFrame(dataNumpy, columns=self.dfHistory.columns)
        # -----------------------------------------------
        # cols = self.dfHistory.shape[1]
        # for i in range(expectedId, cols-1):
        #     dataNumpy[:, i] = dataNumpy[:, i+1]
        # dataNumpy = np.delete(dataNumpy, -1, axis=1)
        # dataFrame = pd.DataFrame(dataNumpy)



        self._createTrainLoader(dataFrame)


        self.b = torch.rand(self.dataset_nStocks, 1).double()
        self.b.requires_grad = True

        # self.b[expectedId] = torch.tensor(0.0, requires_grad=True)
        # self.b.requires_grad = True
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.csv_writer.writerow(
            ["epoch", "day", "index of top", "weights of top", "index of low", "weights of low", "loss"])
        for epoch in range(self.loopTrainEpochs):
            self.optimizer.zero_grad()
            # loss = self._loss(self.b, self.trainloader).cuda(device)
            loss = self._loss(self.b, self.trainloader)#dzw
            loss.backward()
            self.optimizer.step()

            # Normalized and calculate top stocks' index and value
            b_normal = torch.softmax(self.b, dim=0)
            b_list = list(b_normal)
            top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
            last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
            top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
            last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))

            self.csv_writer.writerow(
                [
                    # build a list of
                    epoch,
                    [str(0) + '--' + str(len(self.trainloader) * self.batchsize)],
                    top_index, list(top_value.detach().numpy()),
                    last_index, list(last_value.detach().numpy()),
                    loss.cpu().detach().numpy()
                ])

        b_normal = torch.softmax(self.b, dim=0)
        b_list = list(b_normal)

        top_index = list(map(b_list.index, heapq.nlargest(self.nTopStocks, b_list)))
        last_index = list(map(b_list.index, heapq.nsmallest(self.nLowStocks, b_list)))
        top_value = torch.tensor(heapq.nlargest(self.nTopStocks, b_list))
        last_value = torch.tensor(heapq.nsmallest(self.nLowStocks, b_list))



        y = np.zeros((len(self.b), 1))
        # y[top_index[0]] = 1.0
        for i in top_index:
            y[i] = 1 / self.nTopStocks
        return y


    def outputTopLowData_csv(self, dfHistory, top_index, last_index):
        index = top_index + last_index
        path = os.getcwd() + '/topLowStocksData/' + str(datetime.datetime.now()) + '.csv'

        dfData = dfHistory.iloc[:, index]
        dfData.to_csv(path)



    def _loss(self, x, loader):
        '''
            calculate the average loss between 0 and current batch
        :param x:  weight, self.b

        :param loader:  loader for the dataset from which we compute the loss.
        :return:  loss a number
        '''

        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        y = torch.softmax(x, dim=0)
        dotProds = torch.tensor(0.0)
        reguItem = 0.0
        for j, data in enumerate(loader):
            # if j != len(loader) - 1:
            #     continue
            ksaiBatch = data  # get one batch
            ksaiBatch = ksaiBatch
            Ex = torch.matmul(ksaiBatch, y)
            # Ex:  (batchSize, 1)
            dotProds = dotProds + torch.sum(Ex)  # sum over each samples in each batch

            for r in self.rList:
                # reguItem += (1 / self.batchSize * torch.sum(torch.exp(b - Ex - r)) - alpha_r)
                reguItem += torch.sum(torch.exp(self.b_expectedReturn - Ex - r))
                # torch.sum over batchSize rows of each batch



        # sum1 = dotProds / ((ibatch + 1) * self.batchsize)
        datasize = len(loader) * self.batchsize
        sum1 = dotProds/datasize

        sum2 = reguItem / datasize
        sum3 = 1e+08 * (sum2 - self.alpha_r)

        loss = -sum1 + sum3
        return loss


    def _createTrainLoader(self, dfHistory):
        """

        :param dfHistory: the price data
        :return: return trainLoader
        """

        history_numpy = np.array(dfHistory)
        history_tensor = torch.from_numpy(history_numpy)
        loader = torch.utils.data.DataLoader(history_tensor, batch_size=self.batchsize,
                                             shuffle=True, num_workers=2)
        self.trainloader = loader
