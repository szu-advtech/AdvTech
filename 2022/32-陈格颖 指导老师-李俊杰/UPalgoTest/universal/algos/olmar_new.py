# from ..algo import Algo
from universal.algo import Algo
from universal.result import ListResult
import numpy as np
import pandas as pd
# from .. import tools
from universal import tools
from MyLogger import MyLogger
import os
import datetime
import heapq
from .GetRemainder import GetRemainder, readPKL, olmarBalance

class OLMAR_NEW(Algo):
    """ On-Line Portfolio Selection with Moving Average Reversion

    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10, nstocks=75):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAR_NEW, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps

        # self.logger = MyLogger('olmar_log')
        self.histLen = 0  # yjf.
        self.nstocks = nstocks



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

        #cut dataset
        remainder_dataset = self.cut_dataset(history)[0]

        last_b = olmarBalance(remainder_dataset)  # get last balan     ce of olmar
        # print("last_b", last_b)
        x = remainder_dataset.iloc[history.shape[0] - 1]
        
        # print("%%%%%%last_b", last_b, x)
        x_pred = self.predict(x, remainder_dataset.iloc[-self.window:])

        b = self.update(last_b, x_pred, self.eps)

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

    def cut_dataset(self, dataset):
        x = list(dataset.iloc[-1])
        low_index = map(x.index, heapq.nsmallest(self.nstocks, x))
        low_index = list(low_index)
        item_lists = []
        count = 0
        for item in dataset:
            if count in low_index:
                item_lists.append(item)
            count += 1
        dataset = dataset.drop(item_lists, axis=1)
        return dataset, low_index

    def getEntireBalance(self, b, history):
        """number of b is not equal to dataset,add stock out of b"""
        # df = readPKL(self.filepath)
        nstocks = history.shape[1]
        balance = np.zeros(nstocks)
        itemlists = list(history.iloc[:0])
        bItem = list(b.index)
        # print("-----------", bItem)
        for i in range(len(bItem)):
            for j in range(len(itemlists)):
                if bItem[i] == itemlists[j]:
                    balance[j] = b[i]

        balance = pd.Series(balance, index=itemlists)

        return balance


if __name__ == '__main__':
    result = tools.quickrun(OLMAR())
    res = ListResult([result], ['OLMAR'])
    df = res.to_dataframe()
    df.to_csv('OMLAR_profit.csv')

    result.B.to_csv('OLMAR_balances.csv')
    # t = OLMAR()
    # t.Compare_highBalance_lowBalance(0, 0)