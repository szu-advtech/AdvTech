from universal import tools
import logging

# we would like to see algos progress
logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)

import matplotlib
import matplotlib.pyplot as plt

print('type: ', type(matplotlib.rcParams['savefig.dpi']), 'va: ', matplotlib.rcParams['savefig.dpi'])

from MultiShower import MultiShower
from SimpleSaver import SimpleSaver



class Tester:

    def __init__(self):
        self.data = None
        self.algo = None
        self.result = None
        self.X = None
        self.saver = SimpleSaver()
        self.datasetName = None
        self.NStocks = 0

    def createDataSet(self, datasetName):
        # load data using tools module
        self.data = tools.dataset(datasetName)
        print(self.data)
        self.datasetName = datasetName
        print('data.type: ', type(self.data))
        self.NStocks = self.data.shape[1]
        print(self.data.head())
        print(self.data.shape)


    def runAlgo(self):
        self.result = self.algo.run(self.data)


    def showResult(self, d):

        from universal.algos.crp import CRP
        from universal.algos.olmar import OLMAR
        from universal.algos.bah import BAH
        from universal.algos.bcrp import BCRP
        from universal.algos.pamr import PAMR
        from universal.algos.gwr import GWR
        from universal.algos.rmr import RMR


        result_crp = CRP().run(self.data)
        result_olmar = OLMAR().run(self.data)
        result_bah = BAH().run(self.data)
        result_bcrp = BCRP().run(self.data)
        result_pamr = PAMR().run(self.data)
        result_gwr = GWR().run(self.data)
        result_rmr = RMR().run(self.data)

        ms = MultiShower(self.datasetName)

        for fee in [0]:
            result_crp.fee = fee
            result_olmar.fee = fee
            result_bcrp.fee = fee
            result_pamr.fee = fee
            result_gwr.fee = fee
            result_rmr.fee = fee

            ms.show(
                [

                    result_bah,
                    result_olmar,
                    result_bcrp,
                    result_gwr,
                    result_pamr,
                    result_crp,
                    result_rmr
                ],
                [
                    'BAH',
                    'OLMAR',
                    'BCRP',
                    'GWR',
                    'PAMR',
                    'CRP',
                    'RMR'
                ],
                yLable=self.datasetName + '_Cumulative Wealth=' + str(fee))

            plt.show()

    @staticmethod
    def testSimple():

        datasets = ['djia', 'sp500', 'nyse_o', 'nyse_n', 'tse',  'msci']
        # datasets = ['msci']

        for d in datasets:
            t = Tester()
            t.createDataSet(d)
            t.showResult(d)


if __name__ == '__main__':
    Tester.testSimple()
