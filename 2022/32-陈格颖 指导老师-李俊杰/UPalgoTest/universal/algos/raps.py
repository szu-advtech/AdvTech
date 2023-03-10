#from ..algo import Algo
from universal.algo import Algo
import numpy as np
import pandas as pd
#from .. import tools
from universal import tools


class RAPS(Algo):
    """ Universal Portfolio by Thomas Cover enhanced for "leverage" (instead of just
        taking weights from a simplex, leverage allows us to stretch simplex to
        contain negative positions).

    Reference:
        T. Cover. Universal Portfolios, 1991.
        http://www-isl.stanford.edu/~cover/papers/paper93.pdf
    """

    PRICE = 'ratio'
    REPLACE_MISSING = True


    def __init__(self, eval_points=1E+2, leverage=1.):
        """
        :param eval_points: Number of evaluated points (approximately). Complexity of the
            algorithm is O(time * eval_points * nr_assets**2) because of matrix multiplication.
        :param leverage: Maximum leverage used. leverage == 1 corresponds to simplex,
            leverage == 1/nr_stocks to uniform CRP. leverage > 1 allows negative weights
            in portfolio.
        """
        super(RAPS, self).__init__()
        self.eval_points = eval_points
        self.leverage = leverage
        self.S_total = None

    def init_weights(self, m):
        return np.ones(m) / m


    def init_step(self, X):
        """ Create a mesh on simplex and keep wealth of all strategies. """
        m = X.shape[1]

        # create set of CRPs
        self.W = np.matrix(tools.mc_simplex(m - 1, int(self.eval_points)))
        self.S = np.matrix(np.ones(self.W.shape[0])).T

        # stretch simplex based on leverage (simple calculation yields this)
        leverage = max(self.leverage, 1./m)
        stretch = (leverage - 1./m) / (1. - 1./m)
        self.W = (self.W - 1./m) * stretch + 1./m


    def step(self, x, last_b):
        # calculate new wealth of all CRPs
        self.S = np.multiply(self.S, self.W * np.matrix(x).T)

        if self.S_total is None:
            self.S_total = self.S
        else:
            self.S_total = np.hstack((self.S_total, self.S))

        self.th = []
        for i, data in enumerate(self.S_total):
            df = pd.DataFrame(data)
            expert = df.loc[0]
            #mdd = max(1. - expert / expert.cummax())
            mdd = max(expert.cummax() - expert)
            theta = 1. - mdd
            self.th.append(theta)
            # if self.th is None:
            #     self.th = theta
            # else:
            #     self.th = np.vstack((self.th, theta))
        #print(self.th)

        b = self.W.T * np.matrix(self.th).T

        return b / sum(b)




    def plot_leverage(self, S, leverage=np.linspace(1,10,10), **kwargs):
        """ Plot graph with leverages on x-axis and total wealth on y-axis.
        :param S: Stock prices.
        :param leverage: List of parameters for leverage.
        """
        wealths = []
        for lev in leverage:
            self.leverage = lev
            wealths.append(self.run(S).total_wealth)

        ax = pd.Series(wealths, index=leverage, **kwargs).plot(**kwargs)
        ax.set_xlabel('Leverage')
        ax.set_ylabel('Total Wealth')
        return ax


if __name__ == '__main__':
    data = tools.dataset('djia')
    tools.quickrun(RAPS(), data)
