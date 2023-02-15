from universal.result import ListResult
import matplotlib.pyplot as plt
import datetime


class MultiShower:

    def __init__(self, fileName):

        dtMark = str(datetime.datetime.now()) + '_'
        self.dataSet = fileName
        self.fileName = '/home/m/Desktop/new_result/' + fileName + '_' + dtMark + '.eps'

    def show(self, resultList, algoNameList, yLable='Total Wealth', logy1=True):

        res = ListResult(resultList, algoNameList)
        d = res.to_dataframe()

        # write to file
        # path = 'D:\\holya\\桌面\\前沿技术\\ps算法\\universal\\result_data\\'
        # d.iloc[-1].to_csv(path + 'msci.csv')

        portfolio = d.copy()
        for name in portfolio.columns:
            ax = portfolio[name].plot(figsize=(7, 5), linewidth=1.5, logy=logy1)
        ax.legend()
        ax.set_ylabel(yLable)
        ax.set_xlabel('day')

        # show the last point data
        # algos = ['GWR']
        # for algo in algos:
        #     plt.text(x=portfolio[algo].__len__() - 1, y=portfolio[algo].iloc[-1], s='%.4f' % portfolio[algo].iloc[-1],
        #              ha='center', va='bottom')

        plt.show()
#
