import pandas as pd
import torch
from pandas import Series
import datetime
import csv
import heapq
import numpy as np
from universal.algo import Algo
# from .TopLowStocksSelectors_old import TopLowStocksSelectors
from universal.algos.bcrp import BCRP
from .TopLowStocksSelectors_multiThreads import TopLowStocksSelectors_multiThreads
from .DataLoader import DataLoader
import os
from . import tools

class REBCRP(Algo):
    """ Bay and hold strategy. Buy equal amount of each stock in the beginning and hold them
        forever.  """
    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self):#0.08, 30, 3,3
        """
        :params b: Portfolio weights at start. Default are uniform.
        """
        super(REBCRP, self).__init__()




        # self.omlar = omlar()

    def init_weights(self, m):
        return np.ones(m) / m

    def step(self, x, last_b, history):
        if history.shape[0] % 5 != 0:
            return last_b
        else:
            bcrp = BCRP()
            b = bcrp.weights(history[-5:])
            b = b[-1]

        return b

    def weights(self, X, min_history=None, log_progress=True):
        """

        :param X: raw data. all data divide the first row data.
        :param min_history:
        :param log_progress:
        :return:
        """


        """ Return weights. Call step method to update portfolio sequentially. Subclass
        this method only at your own risk. """
        min_history = self.min_history if min_history is None else min_history

        # init
        B = X.copy() * 0.
        last_b = self.init_weights(X.shape[1])    # call class father init_weights last_b = array([0.333,0.333,...])
        if isinstance(last_b, np.ndarray):
            last_b = pd.Series(last_b, X.columns)

        # use history in step method?
        use_history = self._use_history_step()

        # run algo
        self.init_step(X)
        for t, (_, x) in enumerate(X.iterrows()):
            # save weights
            B.ix[t] = last_b

            # keep initial weights for min_history
            if t < min_history:
                continue

            # trade each `frequency` periods
            if (t + 1) % self.frequency != 0:
                continue

            # predict for t+1
            if use_history:
                history = X.iloc[:t+1]
                last_b = self.step(x, last_b, history)
            else:
                last_b = self.step(x, last_b)

            # convert last_b to suitable format if needed
            if type(last_b) == np.matrix:
                # remove dimension
                last_b = np.squeeze(np.array(last_b))

            # show progress by 10 pcts
            if log_progress:
                tools.log_progress(t, len(X), by=10)
        # B = B[5:]
        swap = B[:5]
        B = B.drop([0, 1, 2, 3, 4])
        B = B.append(swap, ignore_index=True)
        print(type(B))
        print(B)
        return B
