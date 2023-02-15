"""
time: 2021.11
author: dzw
task: select subset of stock according to macd index and using olmar to allocate wealth in subset.
"""
import numpy as np
from universal.algo import Algo
from universal import tools
from universal.subset_tools import GetRemainder
from universal.statistics import Statistics


class OlmarMacd(Algo):
    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, dataset_name, window=5, eps=10, percentage=0.35, top_stock=False, slow_period=26, fast_period=12,
                 signal_period=25):
        """

        :param window:
        :param eps:
        :param percentage: the percentage of subset is selected
        :param top_stock: select the top weight or low weight
        :param slow_period:
        :param fast_period:
        :param signal_period:
        """
        super(OlmarMacd, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.dataset_name = dataset_name
        self.window = window
        self.eps = eps
        self.histLen = 0  # yjf.
        self.percentage = percentage
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.top_stock = top_stock

    def init_weights(self, m):
        return np.ones(m) / m

    def step(self, x, last_b, history):
        """

        :param x: the last row data of history
        :param last_b:
        :param history:
        :return:
        """
        if (history.shape[0] - self.slow_period - self.signal_period + 1) < 0:
            x_pred = self.predict(x, history.iloc[-self.window:])
            b = self.update_new(x_pred)
            return b
        else:
            macder = Statistics(self.dataset_name)
            macd_list = macder.macd(history.shape[0], slow_day=self.slow_period, fast_day=self.fast_period, signal_day=self.signal_period)
            print("macd_list:", macd_list)
            remainder = GetRemainder(macd_list, history, percentage=self.percentage, hightReturn=self.top_stock)
            history = remainder.cutDataset()
            # print("history:", history)
            x = history.iloc[-1]
            x_pred = self.predict(x, history.iloc[-self.window:])
            last_b = self.init_weights(history.shape[1])  # dzw
            # b = self.update(last_b, x_pred, self.eps)
            b = self.update_new(x_pred)
            b = remainder.getEntireBalance(b)
            # print(len(history), b)
            # print("b:", b)
            return b

    # step1
    def step1(self, x, last_b, history):
        if (history.shape[0] - self.slow_period - self.signal_period + 1) < 0:
            b = last_b
            return b
        else:
            macder = Statistics(self.dataset_name)
            macd_list = macder.macd(history.shape[0], slow_day=self.slow_period, fast_day=self.fast_period, signal_day=self.signal_period)
            print("macd_list:", macd_list)
            remainder = GetRemainder(macd_list, history, percentage=self.percentage, hightReturn=self.top_stock)
            history = remainder.cutDataset()
            # print("history:", history)
            x = history.iloc[-1].copy()
            index_list = list(history.columns)
            print("index_list:", index_list)
            print("x:", x)
            for i in index_list:
                x[i] = 1 / history.shape[1]
            b = remainder.getEntireBalance(x)
            return b

    # step2
    def step2(self,x, last_b, history):
        if (history.shape[0] - self.slow_period - self.signal_period + 1) < 0:
            b = last_b
            return b
        else:
            macder = Statistics(self.dataset_name)
            macd_list = macder.macd_normalized(history.shape[0], slow_day=self.slow_period, fast_day=self.fast_period,
                                    signal_day=self.signal_period)
            print("macd_list:", macd_list)
            remainder = GetRemainder(macd_list, history, percentage=self.percentage, hightReturn=self.top_stock)
            history = remainder.cutDataset()
            # print("history:", history)
            x = history.iloc[-1].copy()
            index_list = list(history.columns)
            print("index_list:", index_list)
            print("x:", x)
            for i in index_list:
                x[i] = 1 / history.shape[1]
            b = remainder.getEntireBalance(x)
            return b

    def predict(self, x, history):
        """ Predict returns on next day. """
        return (history / x).mean()
        # calculate the max price among w windows
        # return history.max() / x

    def update_old(self, b, x, eps):
        """

        :param b: weight of last time
        :param x:  predict price
        :param eps: eps = 10
        :return:  weight
        """

        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        b_dot_x = np.dot(b, x)
        gap = (eps - np.dot(b, x))
        x_avg_norm = np.linalg.norm(x - x_mean) ** 2  # 6.28
        gap_n = gap / x_avg_norm  # 6.28

        # lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)
        lam = max(0.0, gap_n)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)
        b = b + lam * (x - x_mean)
        bn = tools.simplex_proj(b)
        return bn

    def update_new(self, x):
        """
        calculate balance according to predicted price
        :param x:
        :return:
        """
        max_index = x.idxmax()
        index_list = list(x.index)
        for i in index_list:
            if i == max_index:
                x[i] = 1
            else:
                x[i] = 0
        return x


if __name__ == '__main__':
    # result = tools.quickrun(OlmarMacd())
    # res = ListResult([result], ['OLMAR'])
    # df = res.to_dataframe()
    # df.to_csv('OMLAR_profit.csv')
    #
    # result.B.to_csv('OLMAR_balances.csv')
    # t = OLMAR()
    # t.Compare_highBalance_lowBalance(0, 0)
    import pandas as pd
    from universal.result import AlgoResult
    from universal.result import ListResult

    datasetList = ['djia', 'tse', 'sp500', 'msci', 'nyse_n', 'hs300']
    for dlist in datasetList:
        path = '/home/l/UPalgoTest4.1/universal/data/' + dlist + '.pkl'
        df_original = pd.read_pickle(path)

        t = OlmarMacd(dlist)
        # df = t._convert_prices(df_original, 'ratio')
        df = t._convert_prices(df_original, 'raw')  # fl olmar用raw格式，传入绝对价格，里面会转换
        B = t.weights(df)
        Return = AlgoResult(t._convert_prices(df_original, 'ratio'), B)

        res = ListResult([Return], ['olmar_macd']).to_dataframe()
        last_return = res.iloc[-1].values[0]

        print(dlist + ":last_return = ", last_return, ",MDD = ", Return.max_drawdown, ",CAR=",
              (Return.annualized_return / Return.max_drawdown))