import pandas as pd
import numpy as np
from ..algo import Algo


class OneStock(Algo):
    def __init__(self, stock_index):
        super(OneStock, self).__init__()
        self.stock_index = stock_index

    def weight(self, S):
        b = np.zeros(S.shape[1])
        b[self.stock_index] = 1
        W = np.repeat([b], S.shape[0], axis=0)
        return W

