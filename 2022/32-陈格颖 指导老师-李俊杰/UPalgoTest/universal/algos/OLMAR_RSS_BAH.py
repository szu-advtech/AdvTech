from universal.algo import Algo
from universal.result import ListResult
import numpy as np
import pandas as pd
# from .. import tools
from universal import tools
import AT_ucb as ap
from MyLogger import MyLogger
import os
import datetime
from .GetRemainder import GetRemainder, readPKL, olmarBalance

class OLMAR_RSS_BAH(Algo):
    """ combinate olmar and rss"""

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, datasetname, window=5, eps=10, percentage=None):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAR_RSS_BAH, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps

        # self.logger = MyLogger('olmar_log')
        self.histLen = 0  # yjf.
        self.datasetname = datasetname
        self.percentage = percentage
        self.batch = window
        self.filepath = "/home/aze/project/UPalgoTest/universal/data/" + datasetname + ".pkl"
        self.history = None
        self.filepath1 = "/home/aze/project/UPalgoTest/ReturnSave/" + datasetname + ".csv"
        self.returnFile = pd.read_csv(self.filepath1)


    def init_weights(self, m):
        return np.ones(m) / m

    def step(self, x, last_b, history):
        """

        :param x: the last row data of history
        :param last_b:
        :param history:
        :return:
        """

        # calculate return prediction
        self.histLen = history.shape[0]
        BAH_return = self.returnFile.iloc[self.histLen-2]["BAH_total_wealth"]
        OLMAR_RSS_BAH = self.returnFile.iloc[self.histLen-2]["OLMAR_total_wealth"]
        if OLMAR_RSS_BAH >= BAH_return:
            dataTool = GetRemainder(history, dataset=self.datasetname, percentage=self.percentage, index=1)
        else:
            dataTool = GetRemainder(history, dataset=self.datasetname, percentage=self.percentage, index=0)

        # dataTool = GetRemainder(history, dataset=self.datasetname, percentage=self.percentage)
        history = dataTool.cutDataset(ndays=self.histLen)
        self.history = history
        # 调用STM
        # b= algo(history)
        b= ap(history)

        # last_b = olmarBalance(history)#get last balan     ce of olmar
        # # print("last_b", last_b)
        # x = history.iloc[history.shape[0]-1]
        # """
        # for i in range(len(last_b)):
        #     if last_b[i] ！= 0
        # """
        # x_pred = self.predict(x, history.iloc[-self.window:])
        # b = self.update(last_b, x_pred, self.eps)
        # # print("&&&bbb%%:", b)
        #
        # # print("######", b)
        # b = self.getEntireBalance(b)  # t the number of stocks equal to dataset


        # print(len(history), b)
        return b


    def predict(self, x, history):
        """ Predict returns on next day. """
        return (history / x).mean()


    def update(self, b, x, eps):
        """

        :param b: weight of last time
        :param x:  predict price
        :param eps: eps = 10
        :return:  weight
        """

        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)

        # print('b: ', b)
        # print('x: ', x)
        b_dot_x = np.dot(b, x)
        # print('b_dot_x: ', b_dot_x)
        gap = (eps - np.dot(b, x))
        # print('gap: ', gap)
        x_avg_norm = np.linalg.norm(x - x_mean)**2
        # print('x_avg_norm: ', x_avg_norm)

        gap_n = gap / x_avg_norm
        # print('gap_n: ', gap_n)


        # lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)
        lam = max(0.0, gap_n)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        # print('----- b: ', b, 'b.type: ', type(b))
        # dtype: float64 b.type:  <class 'pandas.core.series.Series'>
        # print('----- x: ', x, 'x.type: ', type(b))
        b = b + lam * (x - x_mean)

        # print('b: ', b)

        # project it onto simplex
        bn = tools.simplex_proj(b)
        # self.logger.write(str(self.histLen) + '_b_: ' + str(b))
        # self.logger.write(str(self.histLen) + '_bn_: ' + str(bn))

        return bn

    def getEntireBalance(self, b):
        """number of b is not equal to dataset,add stock out of b"""
        df = readPKL(self.filepath)
        nstocks = df.shape[1]
        balance = np.zeros(nstocks)
        itemlists = list(df.iloc[:0])
        bItem = list(b.index)
        # print("-----------", bItem)
        for i in range(len(bItem)):
            for j in range(len(itemlists)):
                if bItem[i] == itemlists[j]:
                    balance[j] = b[i]

        balance = pd.Series(balance, index=itemlists)

        return balance










# if __name__ == '__main__':
    # result = tools.quickrun(OLMAR_RSS())
    # res = ListResult([result], ['OLMAR'])
    # df = res.to_dataframe()
    # df.to_csv('OMLAR_profit.csv')
    #
    # result.B.to_csv('OLMAR_balances.csv')
    # # t = OLMAR()
    # # t.Compare_highBalance_lowBalance(0, 0)
