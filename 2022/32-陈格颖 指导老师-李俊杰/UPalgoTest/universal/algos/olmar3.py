import pandas as pd
import torch
import os
from pandas import Series
import torch.optim as optim
import numpy as np
from universal.algo import Algo
from .PortfolioRisk_weight import PortfolioRisk_weight
from .TopLowStocksSelectors_old import TopLowStocksSelectors
# from .TopLowStocksSelectors import TopLowStocksSelectors
from universal.algos.olmar import OLMAR
from universal.algo import Algo
from MyLogger import MyLogger
import numpy as np
import heapq

from .softmax import Softmax
from .cvxpy_ import CVXPY
from .Wang import Wang

from universal import tools
from MyLogger import MyLogger
# from .GetRemainder import olmarBalance, readPKL, GetRemainder
import datetime
import csv

class OLAMR3(Algo):
    """ Bay and hold strategy. Buy equal amount of each stock in the beginning and hold them
        forever.  """
    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, b_expectedReturn, dataset_nStocks, nTopStocks, nLowStocks, originData, fileName, batchsize=30, window=5, eps=10):
        """
        :params b: Portfolio weights at start. Default are uniform.
        """
        super(OLAMR3, self).__init__(min_history=window)
        self.b_expectedReturn = b_expectedReturn

        self.batchsize = batchsize
        self.dataset_nStocks = dataset_nStocks
        self.t = TopLowStocksSelectors(self.b_expectedReturn, dataset_nStocks, nTopStocks, nLowStocks, originData, self.batchsize)

        self.t_softmax = Softmax()


        self.nTopStocks = nTopStocks
        self.nLowStocks = nLowStocks
        self.fileName = fileName
        self.filePath = os.getcwd() + '/universal/data/' + fileName + '.pkl'

        # the file to save training result:
        self.savefile = os.getcwd() + '/resultStrategy/' + str(datetime.datetime.now()) + '.csv'
        self.file = open(self.savefile, 'w')
        self.csv_writer = csv.writer(self.file)


        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps

        self.logger = MyLogger('olmar_log')
        self.histLen = 0
        self.count = 0


    def init_weights(self, m):
        return np.ones(m) / m

    def step(self, x, last_b, history):
        self.histLen = history.shape[0]

        stocksNum = history.shape[1]
        if self.histLen <= 120:
            b = self.init_weights(stocksNum)
            return b

        orignalData = pd.read_pickle(self.filePath)
        dfhistory = orignalData[:self.histLen]

        nphiostory = np.matrix(dfhistory)
        w = self.t_softmax.step(nphiostory)

        b = pd.Series(w, index=dfhistory.columns)

        return b


    def predict(self, x, history):
        """ Predict returns on next day. """
        # return (history / x).mean()
        return history.max() / x

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


        lam = max(0.0, gap_n)

        lam = min(100000, lam)


        b = b + lam * (x - x_mean)


        bn = tools.simplex_proj(b)
        self.logger.write(str(self.histLen) + '_b_: ' + str(b))
        self.logger.write(str(self.histLen) + '_bn_: ' + str(bn))

        return bn

    def _convert_prices(self, S, method, replace_missing=False):
        """ Convert prices to format suitable for weight or step function.
        Available price types are:
            ratio:  pt / pt_1
            log:    log(pt / pt_1)
            raw:    pt (normalized to start with 1)
        """
        if method == 'raw':
            # normalize prices so that they start with 1.
            r = {}
            for name, s in S.iteritems():
                init_val = s.ix[s.first_valid_index()]
                r[name] = s / init_val
            X = pd.DataFrame(r)

            if replace_missing:
                X.ix[0] = 1.
                X = X.fillna(method='ffill')

            return X

        elif method == 'absolute':
            return S

        elif method in ('ratio', 'log', 'ratio_1'):
            # be careful about NaN values
            X = S / S.shift(1).fillna(method='ffill')
            for name, s in X.iteritems():
                X[name].iloc[s.index.get_loc(s.first_valid_index()) - 1] = 1.

            if replace_missing:
                X = X.fillna(1.)

            # return np.log(X) if method == 'log' else X
            if method == 'log':
                return np.log(X)
            elif method == 'ratio_1':
                return X - 1
            else:
                return X


        else:
            raise ValueError('invalid price conversion method')
