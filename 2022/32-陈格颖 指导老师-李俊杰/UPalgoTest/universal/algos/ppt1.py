import json
import os

from ..algo import Algo
from .. import tools
import numpy as np


class PPT1(Algo):
    """ Bay and hold strategy. Buy equal amount of each stock in the beginning and hold them
    forever.  """

    PRICE_TYPE = 'raw'

    # REPLACE_MISSING = True

    def __init__(self, window=5, eps=10**(-4)):
        """
        :params b: Portfolio weights at start. Default are uniform.
        """



        self.window = window
        self.eps = eps
        self.histLen = 0  # yjf.
        self.gamma = 0.01
        self.eta = 0.005
        self.lam = 0.5
        self.zeta = 500
        self.max_iter = 10**4

        super(PPT1, self).__init__(min_history=0)



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

        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)

        # print(self.histLen, len(b), list(b))

        return b

    def predict(self, x, history):
        """ Predict returns on next day. """
        result = []
        for i in range(history.shape[1]):
            temp =  max(history.iloc[:, i]) / x[i]
            result.append(temp)
        return result

    # def simplex_proj(self, b):
    #     while max(abs(b)) > 10 ** 6:
    #         b = b / 10
    #
    #     u = sorted(b,reverse=True)
    #     su = np.cumsum(u)


    def update(self, b, x, eps):
        """

        :param b: weight of last time
        :param x:  predict price
        :param eps: eps = 100
        :return:  weight
        """
        phi = -1.1 * np.log(x) - 1
        o = 1
        b_o = b
        g_o = b
        rho_o = 0
        vector_1 = np.asarray([1] * len(b))
        I = np.identity(len(b))
        while o < self.max_iter or abs(np.dot(vector_1,b_o)-1) > eps:
            b_o = np.dot(np.linalg.inv((self.lam/self.gamma) * I + self.eta * np.ones((len(b),len(b)))),(self.lam/self.gamma * g_o + (self.eta - rho_o)*vector_1 - phi))
            temp = abs(b_o) - self.gamma * vector_1
            for i in range(len(temp)):
                if temp[i] < 0:
                    temp[i] = 0
            g_o = np.multiply(np.sign(b_o),temp)
            rho_o = rho_o + self.eta * (np.dot(vector_1,b_o)-1)
            o = o + 1



        # project it onto simplex
        bn = tools.simplex_proj(self.zeta * b_o)

        return bn
