import csv
import os
import pandas as pd
import heapq
import matplotlib.pyplot as plt
from universal import tools
from universal import algos
from universal.algo import Algo

import random, datetime
import logging
from MyLogger import MyLogger

import numpy as np

# we would like to see algos progress
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

import matplotlib
import matplotlib.pyplot as plt

print('type: ', type(matplotlib.rcParams['savefig.dpi']), 'va: ', matplotlib.rcParams['savefig.dpi'])

from MultiShower import MultiShower
from SimpleSaver import SimpleSaver

# increase the size of graphs
# matplotlib.rcParams['savefig.dpi'] *= '1.5'

class Tester:

    def __init__(self):
        self.data = None
        self.algo = None
        self.result = None
        self.X = None
        self.logger = MyLogger('PTester_summary')
        self.saver = SimpleSaver()
        self.datasetName = None
        self.NStocks = 0


        # added by yhg.
        self.fileName = os.getcwd() + '/universal/OLMAR_balances/' + str('Balances ') + str(datetime.datetime.now()) + '.csv'

    def createDataSet(self, datasetName):
        # load data using tools module
        self.data = tools.dataset(datasetName)
        # self.data = self.data.iloc[:500]
        self.datasetName = datasetName
        print('data.type: ', type(self.data))

        # plot first three of them as example
        # data.iloc[:,:3].plot()

        self.NStocks = self.data.shape[1]
        print(self.data.head())
        # print(data.tail())
        # plt.show()
        print(self.data.shape)
    def slimDataSet(self, numCols=5):
        # invoked after createDataSet
        n, m = self.data.shape
        # random.randint(1, 10)  # Integer from 1 to 10, endpoints included
        sels = []
        df = pd.DataFrame()
        labels = self.data.columns
        while len(sels) < numCols:
            j = random.randint(0,m-1)
            if j in sels:
                continue
            df[labels[j]] = self.data.iloc[:, j]
            sels.append(j)

        self.data = df
        print('slim_' + self.datasetName + '_', self.data)
        self.NStocks = self.data.shape[1]

    def createRatioX(self):

        PRICE_TYPE = 'ratio'
        self.data = tools.dataset('nyse_o')
        X = Algo._convert_prices(self.data, PRICE_TYPE)
        print('X: ', X)
        mX = X.to_numpy()
        print('shape: ', mX.shape, 'mX:', mX)

    def showNpArray(self):
        arr = self.data.to_numpy()
        print('df.shape: ', self.data.shape, 'arr.shape: ', arr.shape)  # (5651, 36)

    def showNRows(self, index, window):
        rows, cols = self.data.shape
        start = index
        if index > rows:
            start = rows - window
        if index < 0:
            index = 0
        end = start + window
        if end >= cols:
            end = cols - 1
        df = self.data.iloc[range(start, end)]

        print('[ '+str(start)+','+str(end) + ')',  df)

    def showdfIndex(self):
        ind = self.data.index
        print('df.index: ', ind)
        #  Int64Index([   0,    1,    2,    3,    4,    5,    6,    7,    8,    9,
        #             ...
        #             5641, 5642, 5643, 5644, 5645, 5646, 5647, 5648, 5649, 5650],
        #            dtype='int64', length=5651)

    def createAlgo(self, fileName):

        # self.algo = algos.OLMAR()


        # loader = DataLoader(128, True)
        # path = os.getcwd() + "/universal/data/nyse_o_ratio.pkl"
        # trainloader, col = loader.getLoaderFromPickle(path)
        # t = PortfolioRisk(0.9, trainloader, trainloader, col)
        # t.visualOriginalData(path)
        # self.algo = algos.BAH_batch(2, 2, t)


        # set algo parameters
        dataset_nStocks = self.data.shape[1]
        nTopStocks = 3
        nLowStocks = 3
        # self.algo = algos.RSS(0.03, dataset_nStocks, nTopStocks, nLowStocks, self.data)  # create OLMAR2 algo.
        # self.algo_softmax = algos.RUN_SOFTMAX(0.6, dataset_nStocks, nTopStocks, nLowStocks, self.data, fileName)
        # self.algo_wang = algos.RUN_WANG(0.6, dataset_nStocks, nTopStocks, nLowStocks, self.data, fileName)
        # self.algo_cvxpy = algos.RUN_CVXPY(0.6, dataset_nStocks, nTopStocks, nLowStocks, self.data, fileName)
        # self.algo_v = algos.RUN_V(0.6, dataset_nStocks, nTopStocks, nLowStocks, self.data, fileName)

        self.algo_ppt = algos.PPT()
        # self.algo_ek = algos.AutoAlgoSwitch(fileName)
        # self.algo_ek_ucb = algos.AutoAlgoSwitch_U(fileName)


        #


        # self.algo_spolc = algos.SPOLC(fileName)
        # self.algo = algos.AutoAlgo(fileName)  # max_up
        # self.algo_min = algos.AutoAlgoMin(fileName)#min_up
        # self.algo_min_down = algos.AutoAlgoMinDown(fileName)#min_down
        # self.algo_max_down = algos.AutoAlgoMaxDown(fileName)#min_down
        # #
        # self.algo_f = algos.AutoAlgo_F(fileName)  # max_up_f
        # self.algo_min_f = algos.AutoAlgoMin_F(fileName)  # min_up_f
        # self.algo_min_down_f = algos.AutoAlgoMinDown_F(fileName)  # min_down_f
        # self.algo_max_down_f = algos.AutoAlgoMaxDown_F(fileName)  # min_down_f
        # self.algo_switch_handmade = algos.AutoAlgoSwitchHandmade(fileName)
        # self.algo_gwr = algos.GWR(fileName)
        self.algo_olmar = algos.OLMAR()
        # self.algo_bah = algos.BAH()
        self.algo_bcrp = algos.BCRP()
        # self.algo_pamr = algos.PAMR()
        # self.algo_lsrt = algos.LSRT()
        self.algo_rmr = algos.RMR()



        # self.algo = algos.RSS_old(0.03, dataset_nStocks, nTopStocks, nLowStocks)
        # return self.algo, self.algo_max_down, self.algo_min_down, self.algo_min, self.algo_ek
        # return self.algo, self.algo_ek
    def runAlgo(self):
        self.result_ppt = self.algo_ppt.run(self.data)
        # self.result_spolc = self.algo_spolc.run(self.data)
        self.result_olmar = self.algo_olmar.run(self.data)
        # self.result_ek = self.algo_ek.run(self.data)
        # self.result_ek_ucb = self.algo_ek_ucb.run(self.data)

        # self.result_gwr = self.algo_gwr.run(self.data)
        self.result_bcrp = self.algo_bcrp.run(self.data)
        # self.result_bah = self.algo_bah.run(self.data)
        # self.result_pamr = self.algo_pamr.run(self.data)
        # self.result_lsrt = self.algo_lsrt.run(self.data)
        self.result_rmr = self.algo_rmr.run(self.data)

        # self.result = self.algo.run(self.data)
        # self.result_min = self.algo_min.run(self.data)
        # self.result_min_down = self.algo_min_down.run(self.data)
        # self.result_max_down = self.algo_max_down.run(self.data)

        # self.result_f = self.algo_f.run(self.data)
        # self.result_min_f = self.algo_min_f.run(self.data)
        # self.result_min_down_f = self.algo_min_down_f.run(self.data)
        # self.result_max_down_f = self.algo_max_down_f.run(self.data)
        # self.result_switch_handmade = self.algo_switch_handmade.run(self.data)


        # self.result_wang = self.algo_wang.run(self.data)
        # self.result_cvxpy = self.algo_cvxpy.run(self.data)

    def getDataSetNameWithDT(self):
        tMark = str(datetime.datetime.now())
        return self.datasetName
        # return self.datasetName + '_' + str(self.NStocks) + '_' + tMark + '_'

    def showResult(self, d):

        portfolio_label = 'RSS'
        # portfolio_label = 'OLMAR_old'
        from universal.algos.bah import BAH
        from universal.algos.olmar import OLMAR
        from universal.algos.bcrp import BCRP
        from universal.algos.anticor import Anticor
        from universal.algos.bcrp_batch import BCRP_batch
        # from universal.algos.softmax import Softmax
        from universal.algos.spolc import SPOLC

        # from universal.algos.eg import EG
        # from universal.algos.corn import CORN
        # from universal.algos.ons import ONS
        # from universal.algos.pamr import PAMR

        # path = './resultSave/' + self.getDataSetNameWithDT()

        # print(self.datasetName + '_WFM_' +self.result.summary())
        # self.result.save(path)   ###
        # self.logger.write(self.datasetName + '_RSS_' + self.result.summary())    ###
        # result_bah = BAH().run(self.data)
        # result_bcrp_batch = BCRP_batch(self.data).run(self.data)
        result_bcrp = BCRP().run(self.data)

        result_olmr = OLMAR().run(self.data)
        # result_anticor = Anticor().run(self.data)
        # result_spolc = SPOLC(d).run(self.data)

        # result_olmr_macd = algos.OlmarMacd(d).run(self.data)

        # result_bcrp.save()
        # ###

        ms = MultiShower(self.datasetName)
        # ms = MultiShower(self.getDataSetNameWithDT() + '_Result_')
        # result_up     = UP().run(self.data)
        for fee in [0.0]:
            self.result_ppt.fee = fee
            # self.result_spolc.fee = fee
            # result_olmr_macd.fee = fee
            # self.result_gwr.fee = fee
            self.result_olmar.fee = fee
            # self.result_ek.fee = fee
            # self.result_ek_ucb.fee = fee
            # self.result.fee = fee
            # self.result_min.fee = fee
            # self.result_max_down.fee = fee
            # self.result_min_down.fee = fee
            # self.result_f.fee = fee
            # self.result_min_f.fee = fee
            # self.result_max_down_f.fee = fee
            # self.result_min_down_f.fee = fee
            # self.result_switch_handmade.fee = fee

            # self.result_wang.fee = fee
            # self.result_bah.fee = fee
            self.result_bcrp.fee = fee
            # self.result_pamr.fee = fee
            #
            self.result_rmr.fee = fee
            # self.result_lsrt.fee = fee
            # result_olmr.fee = fee
            # result_bcrp.fee = fee
            # self.result_gwr.fee = fee

            # result_anticor.fee = fee
            # result_spolc.fee = fee
            # ms.show([result_bah, result_bcrp_batch,
            #          result_olmr, result_anticor, result_softmax
            #          ],
            #         ['BAH', 'BCRP_barch', 'OLMAR', 'ANTICOR',
            #          portfolio_label, 'SOFTMAX'],
            #         yLable=self.datasetName + ' Total Wealth=' + str(fee))
            # ms.show([self.result_ek, self.result_gwr, result_bah, self.result_ppt, self.result_spolc, result_olmr, self.result],
            #         ['S4', 'GWR', 'BAH', 'PPT', 'SPLOC', 'OLMAR', 'MAX_UP'],
            #         yLable=self.datasetName + ' Total Wealth=' + str(fee))
            # ms.show(
            #     [self.result, self.result_min, self.result_max_down, self.result_min_down, self.result_ek],
            #     ['MAXUP', 'MINUP', 'MAXDOWN', 'MINDOWN', 'S4'],
            #     yLable=self.datasetName.upper() + '  Cumulative Wealth')
            # ms.show(
            #     [self.result_spolc,self.result_gwr,self.result_olmar,self.result_bah,self.result_bcrp, self.result_ppt, self.result_rmr, self.result_pamr, self.result_ek, self.result_ek_ucb],
            #     ['SPLOC', 'GWR', 'OLMAR', 'BAH', 'BCRP', 'PPT', 'RMR', 'PAMR', 'S4', 'S4_UCB'],
            #     yLable=self.datasetName.upper() + '  Cumulative Wealth')
            ms.show(
                [self.result_ppt,self.result_olmar,self.result_bcrp,self.result_rmr],
                ['PPT',"OLMAR","BCRP","RMR"],
                yLable=self.datasetName.upper() + '  Cumulative Wealth')
            # ms.show(
            #     [self.result, self.result_ek],['max_up', 'S4'],yLable=self.datasetName + ' Total Wealth=' + str(fee))


            plt.show()
            # plt.savefig('/home/m/Desktop/switch_result_new/' + d + '.eps')
            #
            # self.logger.write(self.datasetName + '_BAH_' + str(fee) + '_' + self.result_bah.summary())
            # self.logger.write(self.datasetName + '_S4_' + str(fee) + '_' + self.result_ek.summary())
            # self.logger.write(self.datasetName + '_S4_UCB_' + str(fee) + '_' + self.result_ek_ucb.summary())
            # # # self.logger.write(self.datasetName + portfolio_label + str(fee) + '_' + self.result.summary())   ###
            # self.logger.write(self.datasetName + '_OLMAR_' + str(fee) + '_' + self.result_olmar.summary())
            # # # self.logger.write(self.datasetName + '_ANTICOR_' + str(fee) + '_' + result_anticor.summary())
            # # # self.logger.write(self.datasetName + '_SOFTMAX_' + str(fee) + '_' + result_softmax.summary())
            # #
            # self.logger.write(self.datasetName + '_SPOLC_' + str(fee) + '_' + self.result_spolc.summary())
            # self.logger.write(self.datasetName + '_GWR_' + str(fee) + '_' + self.result_gwr.summary())
            # self.logger.write(self.datasetName + '_BCRP_' + str(fee) + '_' + self.result_bcrp.summary())
            # self.logger.write(self.datasetName + '_PPT_' + str(fee) + '_' + self.result_ppt.summary())
            # self.logger.write(self.datasetName + '_PAMR_' + str(fee) + '_' + self.result_pamr.summary())
            # self.logger.write(self.datasetName + '_RMR_' + str(fee) + '_' + self.result_rmr.summary())



    def getLowMDD(self):
        from universal.algos.bah import BAH
        path = os.getcwd() + '/universal/data/' + self.dataFile + '.pkl'
        data = pd.read_pickle(path)
        col = data.shap
        fig1 = plt.gcf()
        # switch_time_auto = [151]
        # switch_time = []
        # for i in switch_time:e[1]

        # file = open(savefile, 'w')
        ARList = []
        SharpeList = []
        MDDList = []
        for k in range(0, col):
            one = np.zeros(col)
            one[k] = 1
            result_bah = BAH(one).run(data)
            result_bah.fee = 0.0

            # file.write('ARL:' + str(result_bah.annualized_return) + '      Sharp:' + str(result_bah.sharpe) + '      MDD:' + str(result_bah.max_drawdown) + '\n')

            ARList.append(result_bah.annualized_return)
            SharpeList.append(result_bah.sharpe)
            MDDList.append(result_bah.max_drawdown)
            # stockPer.append([result_bah.annualized_return, result_bah.sharpe, result_bah.max_drawdown])

        min_MDD_index = list(map(MDDList.index, heapq.nsmallest(self.numOfSelectedStocks, MDDList)))
        return min_MDD_index

    def olmarTopLowWeight_tocsv(self, dataFile, numOfSelectedStocks=3):

        from universal.algos.olmar import OLMAR
        from universal.algos.bcrp_batch import BCRP_batch
        from universal.result import ListResult
        self.dataFile = dataFile
        self.numOfSelectedStocks = numOfSelectedStocks
        # min_MDD_index = self.getLowMDD()
        # print("stocks index of the lowest MDD ", min_MDD_index)

        ds = tools.dataset(self.dataFile)    # ds = tools.dataset('nyse_o)

        path = os.getcwd() + '/universal/data/nyse_o.pkl'
        df = pd.read_pickle(path)

        # result = tools.quickrun(OLMAR(), ds)
        result = tools.quickrun(BCRP_batch(df), ds)
        res = ListResult([result], ['OLMAR'])
        # df = res.to_dataframe()
        # df.to_csv('OMLAR_profit.csv')

        result.B.to_csv(self.fileName)
        balancesData = pd.read_csv(self.fileName)

        path = os.getcwd() + '/olmarChooseStocksWeight/chooseTopLowStocksData.csv'
        file = open(path, 'w')
        csv_write = csv.writer(file)
        csv_write.writerow(['day', 'index of top stocks', 'weight of top stocks', 'index of low stocks', 'weight of low stocks'])
        # read data by line
        for index in balancesData.index:
            dfData = balancesData.iloc[index]
            data_list = list(dfData)
            data_list.remove(data_list[0])
            top_indexs = list(map(data_list.index, heapq.nlargest(self.numOfSelectedStocks,data_list)))
            top_values = list(heapq.nlargest(self.numOfSelectedStocks, data_list))
            low_indexs = list(map(data_list.index, heapq.nsmallest(self.numOfSelectedStocks, data_list)))
            low_values = list(heapq.nsmallest(self.numOfSelectedStocks, data_list))
            csv_write.writerow([index, top_indexs, top_values, low_indexs, low_values])


        # calculate balances file numbers of value >= 0.5
        res = (balancesData >= 0.5).astype(int).sum(axis=0)
        couters = []
        i = 0
        for data in res[1:]:
            couters.append([data, i])
            i = i + 1
        couters.sort(reverse=True)

        # csv_write.writerow([0, 0, 0, 0, couters])
        print(couters)
        # the highest numbers index and values of value >= 0.5
        high_stocks_index = list(map(couters.index, heapq.nlargest(self.numOfSelectedStocks, couters)))
        high_stocks_value = heapq.nlargest(3, couters)
        print('The highest index and numbers of value >= 0.5:', high_stocks_index, high_stocks_value)

        # the lowest numbers index and values of value >= 0.5
        numOfLowStocks = self.numOfSelectedStocks
        low_stocks_index = list(map(couters.index, heapq.nsmallest(numOfLowStocks, couters)))
        low_stocks_value = heapq.nsmallest(numOfLowStocks, couters)
        print('The lowest index and numbers of value >= 0.5:', low_stocks_index, low_stocks_value)

    @staticmethod
    def testSimple():


        # datasets = ['hs300', 'djia', 'stoxx_raw', 'fof', 'msci',  'tse']
        # datasets = ['djia', 'hs300', 'msci', 'nyse_o', 'STOXX50', 'fof']
        # datasets = ['msci', 'hs300', 'nyse_n']
        datasets = ['djia']
        # datasets = ['new_data','20220308181808']

        # datasets = ['djia_return',  'msci_return', 'hs300_return', 'tse_return', 'sp500_return', 'nyse_n_return']



        for d in datasets:
            t = Tester()
            t.createDataSet(d)
            t.createAlgo(d)
            t.runAlgo()
            t.showResult(d)

    @staticmethod
    def convertDataset():

        data = tools.dataset('FTSE100_data')
        bah = algos.BAH(b=None)  # change to your weights
        # convert prices to proper format
        # print('------------raw: each record divided the first one-----------')
        # X = bah._convert_prices(data, 'raw')
        # print(X.head(5))
        #
        # print('------------absolute-: original data-----------')
        # X = bah._convert_prices(data, 'absolute')
        # print(X.head(5))
        #
        # print('----------ratio--each record divided by the previous one---------------')
        X = bah._convert_prices(data, 'ratio')
        print(X.head(5))
        X.to_pickle('./universal/data/tse_ratio.csv')


    @staticmethod
    def testOLMAR():
        from universal.algos.olmar import OLMAR
        from universal.result import ListResult
        ds = tools.dataset('sp500')
        result = tools.quickrun(OLMAR(), ds)
        res = ListResult([result], ['OLMAR'])
        df = res.to_dataframe()

        ds.to_csv('nyse_o.csv')
        df.to_csv('OMLAR_profit.csv')
        result.B.to_csv('OLMAR_balances.csv')

if __name__ == '__main__':
    Tester.testSimple()
    # Tester.convertDataset()
    # Tester.testOLMAR()

    # t = Tester()
    # t.olmarTopLowWeight_tocsv('nyse_o')