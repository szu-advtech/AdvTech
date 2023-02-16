import json
import os

from ..algo import Algo
from .. import tools
import numpy as np


class PPT(Algo):
    """ Bay and hold strategy. Buy equal amount of each stock in the beginning and hold them
    forever.  """

    PRICE_TYPE = 'raw'

    # REPLACE_MISSING = True

    def __init__(self, window=5, eps=100):
        """
        :params b: Portfolio weights at start. Default are uniform.
        """



        self.window = window
        self.eps = eps
        self.histLen = 0  # yjf.

        super(PPT, self).__init__(min_history=self.window)



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
            temp = max(history.iloc[:, i]) / x[i]
            result.append(temp)
        return result

    def update(self, b, x, eps):
        """

        :param b: weight of last time
        :param x:  predict price
        :param eps: eps = 100
        :return:  weight
        """

        identity_matrix = np.eye(len(b)) - 1 / len(b)
        x_hat = []

        count_x_hat = 0

        for i in range(len(b)):
            temp = np.dot(identity_matrix[i], x)
            # print(type(temp))
            x_hat.append(temp)
            # print(np.around(np.dot(identity_matrix[i], x),3))
            count_x_hat = count_x_hat + abs(temp)
        # print(x_hat)

        x_hat_norm = np.linalg.norm(x_hat)
        # update portfolio
        for i in range(len(x_hat)):
            x_hat[i] = x_hat[i] * eps / x_hat_norm

        if count_x_hat == 0:
            b = b
        else:
            for i in range(len(x_hat)):
                b[i] = b[i] + x_hat[i]

        # project it onto simplex
        bn = tools.simplex_proj(b)

        return bn
