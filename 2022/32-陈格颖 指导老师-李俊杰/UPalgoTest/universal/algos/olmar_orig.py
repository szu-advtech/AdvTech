from ..algo import Algo
import numpy as np
import pandas as pd
from .. import tools


class OLMAR(Algo):
    """ On-Line Portfolio Selection with Moving Average Reversion

    Reference:
        B. Li and S. C. H. Hoi.
        On-line portfolio selection with moving average reversion, 2012.
        http://icml.cc/2012/papers/168.pdf
    """

    PRICE_TYPE = 'raw'
    REPLACE_MISSING = True

    def __init__(self, window=5, eps=10):
        """
        :param window: Lookback window.
        :param eps: Constraint on return for new weights on last price (average of prices).
            x * w >= eps for new weights w.
        """

        super(OLMAR, self).__init__(min_history=window)

        # input check
        if window < 2:
            raise ValueError('window parameter must be >=3')
        if eps < 1:
            raise ValueError('epsilon parameter must be >=1')

        self.window = window
        self.eps = eps


    def init_weights(self, m):
        return np.ones(m) / m


    def step(self, x, last_b, history):

        print('last_b: ', last_b)

        # calculate return prediction
        # print('OLMAR_history: ', history)
        # print('OLMAR_last_b: ', last_b)
        # print('OLRM__history: ', history)
        # OLRM__history...raw
        # 0  1.000000  1.000000  1.000000  ...  1.000000  1.000000  1.000000
        # 1  1.014930  1.040360  0.989050  ...  1.009580  0.990880  1.002480
        # 2  1.014930  1.015693  0.967152  ...  1.009580  1.018238  0.999994
        # 3  1.039806  1.022417  0.992705  ...  1.011498  1.021283  1.022324
        # 4  1.072040  1.006723  0.996358  ...  1.026822  1.033436  1.032250
        # 5  1.077025  1.022418  0.985408  ...  1.017242  1.009119  1.034728

        # OLRM_history...ratio
        # 0  1.00000  1.00000  1.00000  1.00000  ...  1.00000  1.00000  1.00000  1.00000
        # 1  1.01493  1.04036  0.98905  0.99490  ...  1.01938  1.00958  0.99088  1.00248

        # 2  1.00000  0.97629  0.97786  0.99744  ...  0.97338  1.00000  1.02761  0.99752
        # 3  1.02451  1.00662  1.02642  1.00257  ...  1.00000  1.00190  1.00299  1.02233
        # 4  1.03100  0.98465  1.00368  1.00513  ...  1.01563  1.01515  1.01190  1.00971
        # 5  1.00465  1.01559  0.98901  1.01020  ...  1.01154  0.99067  0.97647  1.00240

        # 6  1.00694  0.99781  1.00000  1.00758  ...  1.02281  1.00377  1.01807  1.00000

        x_pred = self.predict(x, history.iloc[-self.window:])
        b = self.update(last_b, x_pred, self.eps)

        # print('OLMAR_predicted_b: ', b)
        # every sparse, only investing 1 asset
        return b


    def predict(self, x, history):
        """ Predict returns on next day. """
        # print('____OLRM__history__forPredict: ', history)
        # print('____OLMAR___currentPriceRelative: ', x)
        # ____OLRM__history__forPredict:
        # 1  1.014930  1.040360  0.989050  ...  1.009580  0.990880  1.002480
        # 2  1.014930  1.015693  0.967152  ...  1.009580  1.018238  0.999994
        # 3  1.039806  1.022417  0.992705  ...  1.011498  1.021283  1.022324
        # 4  1.072040  1.006723  0.996358  ...  1.026822  1.033436  1.032250
        # 5  1.077025  1.022418  0.985408  ...  1.017242  1.009119  1.034728

        # OLRM_history_forPredict...ratio
        # 2  1.00000  0.97629  0.97786  0.99744  ...  0.97338  1.00000  1.02761  0.99752
        # 3  1.02451  1.00662  1.02642  1.00257  ...  1.00000  1.00190  1.00299  1.02233
        # 4  1.03100  0.98465  1.00368  1.00513  ...  1.01563  1.01515  1.01190  1.00971
        # 5  1.00465  1.01559  0.98901  1.01020  ...  1.01154  0.99067  0.97647  1.00240

        #

        return (history / x).mean()


    def update(self, b, x, eps):
        """ Update portfolio weights to satisfy constraint b * x >= eps
        and minimize distance to previous weights. """
        x_mean = np.mean(x)
        lam = max(0., (eps - np.dot(b, x)) / np.linalg.norm(x - x_mean)**2)

        # limit lambda to avoid numerical problems
        lam = min(100000, lam)

        # update portfolio
        b = b + lam * (x - x_mean)

        # project it onto simplex
        return tools.simplex_proj(b)


if __name__ == '__main__':
    tools.quickrun(OLMAR())

