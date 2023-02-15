# from ..algo import Algo
from universal.algo import Algo
from universal.result import ListResult
import numpy as np
import pandas as pd
from universal import tools
from MyLogger import MyLogger
import os
import datetime


class GWR(Algo):
    """ Gaussian Weighting reversion Strategy for Accurate Online Portfolio Selection

    Reference:
        Xia Cai and Zekun Ye
        Gaussian Weighting reversion Strategy for Accurate Online Portfolio Selection, 2019.
    """

    # Original price
    PRICE_TYPE = 'absolute'

    # if true, replace missing values by last values
    REPLACE_MISSING = True

    def __init__(self, tor=2.8, epsilon=0.005, delta=50):
        """
        :param tor: a parameter of Gaussian function
        :param epsilon: is used to control the weighted range
        :param delta: Constraint on return for new weights on last price (average of prices).
            x * w >= delta for new weights w.
        """

        """add self-learning mechanism for tor"""
        # form the weighted window size
        window = round(np.power(-2 * (tor ** 2) * np.log(epsilon), 0.5))
        super(GWR, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if delta < 1:
            raise ValueError('delta parameter must be >=1')

        self.window = int(window)
        self.delta = delta
        self.tor = tor
        self.epsilon = epsilon
        self.histLen = 0  # yjf.

    def init_weights(self, m):
        return np.ones(m) / m

    def step(self, x, last_b, history):
        """

        :param x: the last row data of history
        :param last_b: the last portfolio strategy
        :param history: the history data sequence
        :return:
        """
        # calculate relative price prediction on next day
        self.histLen = history.shape[0]
        history1 = history.iloc[-self.window:]
        history2 = history.iloc[-self.window - 1:-1]
        x_pred = self.predict(x, history1, history2)

        # update portfolio
        b = self.update(last_b, x_pred, self.delta)

        return b

    def gaussian_function(self):
        """
        :return: the vector of the weight
        """
        gaussian_weight = []
        for i in range(-self.window + 1, 1):
            # i = [-8,-7,-6,-5,-4,-3,-2,-1,0]
            gaussian_weight.append(np.exp(-(1 - i) ** 2 / (2 * np.power(self.tor, 2))))

        return np.array(gaussian_weight)

    def pp1(self, history):
        weight = self.gaussian_function()
        return np.dot(history.T, weight) / weight.sum()

    def pp2(self, history1, history2):
        weight = self.gaussian_function()
        # deep copy,change history but history1 not change
        history = history1.copy()
        # the actual closing prize of pt replace by the predict one
        history.iloc[-1] = self.pp1(history2)

        return np.dot(history.T, weight) / weight.sum()

    def predict(self, x, history1, history2):
        """ Predict returns on next day. """
        # ppt1(t,t+1)
        ppp1 = self.pp1(history1)
        # ppt1(t-1,t)
        ppp2 = self.pp2(history1, history2)
        # the final prediction of p_t+1
        predict_price = (ppp1 + ppp2) / 2

        return predict_price / x

    def update(self, b, x, delta):
        """
        :param b: weight of last time
        :param x:  predict price
        :param delta: delta = 50
        :return:  weight
        """

        """ Update portfolio weights to satisfy constraint b * x >= delta
        and minimize distance to previous weights. """

        x_mean = np.mean(x)

        gap = (delta - np.dot(b, x))
        x_avg_norm = np.linalg.norm(x - x_mean) ** 2

        gap_n = gap / x_avg_norm

        lam = max(0.0, gap_n)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)
        # print("------------b", b)

        # project it onto simplex
        bn = tools.simplex_proj(b)

        return bn
