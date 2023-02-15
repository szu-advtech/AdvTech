
from universal.result import ListResult
import pandas as pd
import matplotlib.pyplot as plt
import datetime

class MultiShower:

    def __init__(self, fileName):

        dtMark = str(datetime.datetime.now()) + '_'
        self.dataSet = fileName
        self.fileName = '/home/m/Desktop/new_result/' + fileName + '_' + dtMark + '.eps'

    def show(self, resultList, algoNameList, yLable='Total Wealth', logy1=True):
        # symbol = {"MAXUP":'+', "MINUP":'_', "MAXDOWN":'+', "MINDOWN":'_'}
        res = ListResult(resultList, algoNameList)
        d = res.to_dataframe()
        portfolio = d.copy()

        # line_color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728"]
        line_color = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd","#8c564b","#e377c2","#7f7f7f", "#bcbd22", "#081843","#d62728"]
        i = 0
        # algos_name = ['MAXUP', 'MINUP', 'MAXDOWN', 'MINDOWN' ]
        for name in portfolio.columns:
            # if name == "S4":
            #     ax = portfolio[name].plot(linewidth=1.5, logy=logy1, linestyle='solid', label=name, color=line_color[i])
            # elif name == "BAH":
            #     ax = portfolio[name].plot(linewidth=1.5, logy=logy1, linestyle='solid', label=name, color=line_color[i])
            # else:
                # ax = portfolio[name].plot(linestyle = 'dashed',linewidth=0.8, logy=logy1 , label=name)
            ax = portfolio[name].plot(linewidth=1.5, logy=logy1, linestyle='solid', label=name, color=line_color[i])
            i = i+1


            # if name in symbol.keys():
            #     if name == "MAXUP" or name == "MINUP":
            #         ax = portfolio[name].plot(linestyle = 'solid',linewidth=0.5, logy=logy1)
            #     else:
            #         ax = portfolio[name].plot(linestyle = 'dashed',linewidth=0.5, logy=logy1)
            # if name == "S4":
            #     ax = portfolio[name].plot(linewidth=1.5, logy=logy1, color=line_color[i], linestyle='solid', label=name)
            # else:
            #     ax = portfolio[name].plot(linewidth=0.5, logy=logy1, color=line_color[i], linestyle='dashed', label=name)
            # i = i+1

        # begin_times = [[50, 101, 152, 363, 428, 501, 601, 751, 832, 907, 1103,1584, 1885, 1959, 2175, 2406],
        #                [0, 80, 301, 440, 548, 751, 819, 1052, 1302, 1651, 1853, 2007, 2206, 2450],
        #                [252, 410, 451, 730, 882, 1151, 1555, 1802, 1928],
        #                [40, 138, 188, 2156]]
        # end_times = [[80, 138,188, 410, 440, 548, 730, 819, 882, 1052, 1151, 1651, 1928, 2007, 2206, 2450],
        #              [40, 101, 363, 451, 601, 767, 832, 1103, 1555, 1802, 1885, 2156, 2406, 2548],
        #              [301, 428, 501, 751, 907, 1302, 1584, 1853, 1959],
        #              [50, 152, 252, 2175]]

        # hs300
        # begin_times = [[20,49,180,258,401,417],
        #                [0,109,175,191,353,411],
        #                [38,261,407],
        #                [21,157,177]]
        # end_times = [[21,109,191,261,407,420],
        #              [20,157,177,258,401,417],
        #              [49,353,411],
        #              [38,175,180]]
        # for i in range(1,5):
        #     self.temp = []
        #     for j in range(len(begin_times[i-1])):
        #         self.temp.append(portfolio[algos_name[i-1]][begin_times[i-1][j]:end_times[i-1][j] + 1])
        #     for t in self.temp:
        #         ax = t.plot(linewidth=1.5, logy=logy1, color=line_color[i-1], linestyle='solid', label='')
        # temp = portfolio["MAXUP"][50:81]
        # while i <= len(begin_times[0])-1:
        #     temp = pd.concat([temp, portfolio["MAXUP"][begin_times[0][i]:end_times[0][i] + 1]], axis=0)
        #     i = i + 1


        ax.set_ylabel(yLable)
        ax.set_xlabel('day')

        plt.xlim(0, d.shape[0]+d.shape[0]/25)

        # fig1 = plt.gcf()
        if self.dataSet == "djia":
            switch_time_auto = [20, 28, 31, 61, 151, 152, 191, 201, 251, 306, 412, 413]
            # switch_time_auto = []
            # switch_time = [105,360]
            # switch_time = [197, 369, 370]
        elif self.dataSet == "fof":
            # switch_time_auto = [121, 172, 174, 189, 190, 194, 196, 833, 838, 839, 841, 848, 851, 853, 856, 887, 893, 897, 903]
            # switch_time = [92]
            switch_time_auto = [20, 28, 51, 101, 151, 153, 156, 251, 260, 264, 274, 275, 301, 306, 307, 346, 351, 355, 357, 363, 377, 380, 391, 395, 401, 402, 403, 405, 411, 418, 424, 448, 451, 457, 501, 514, 522, 536, 539, 547, 551, 557, 558, 559, 561, 643, 646, 672, 735, 736, 1001, 1007, 1051, 1086, 1087, 1101, 1103]

        elif self.dataSet == "stoxx":
            # switch_time_auto = [121, 189, 190, 196, 461, 464, 465, 466, 467, 479, 508, 511, 515, 722, 725, 726, 735, 739, 756, 949, 952, 953, 956, 959, 960, 963, 964, 1086, 1093, 1100, 1101, 1106, 1107, 1112, 1115, 1116, 1162, 1190, 1192, 1193, 1205, 1212, 1213, 1217, 1218, 1220, 1221]
            # switch_time = [610,750,927,1050]
            # switch_time_auto = [20, 28, 33, 36, 39, 40, 51, 52, 81, 101, 138, 151, 152, 188, 191, 201, 203, 253, 301, 351, 363, 410, 428, 440, 451, 501, 502, 548, 553, 601, 730, 733, 738, 742, 751, 767, 774, 819, 832, 833, 846, 857, 872, 875, 878, 882, 894, 898, 901, 907, 962, 965, 1052, 1101, 1104, 1151, 1201, 1203, 1208, 1302, 1303, 1555, 1584, 1599, 1601, 1651, 1690, 1701, 1705, 1710, 1748, 1749, 1751, 1752, 1757, 1801, 1802, 1811, 1812, 1832, 1853, 1885, 1928, 1951, 1959, 1963, 2007, 2156, 2175, 2206, 2301, 2306, 2314, 2318, 2322, 2406, 2451, 2456, 2501, 2515]
              switch_time_auto = [601, 1103, 1302, 2006]
            # switch_time = []
        elif self.dataSet == "hs300":
            # switch_time_auto = [20, 21, 51, 67, 101, 107, 157, 175, 177, 180, 191, 258, 261, 263, 291, 353, 401, 407, 412, 417, 420]
            switch_time_auto = [20, 21, 22, 38, 49, 101, 102, 110, 157, 175, 177, 180, 191, 258, 261, 353, 401, 407, 411, 417]
            switch_time_auto = [49, 157, 191, 261, 353]
            # switch_time_auto = [121, 259, 260, 262, 292, 361, 366, 367, 368, 381, 384, 398, 399, 402]
            # switch_time = [220,316]
        elif self.dataSet == "msci":
            switch_time_auto = [20, 21, 51, 58, 64, 66, 67, 84, 101, 161, 162, 205, 304, 317, 321, 323, 671, 672]
            # switch_time_auto = [229, 235, 245, 254, 292, 297, 475, 504, 506, 509, 510, 512, 677, 692, 812, 815, 826, 829, 830, 831, 834, 838, 841, 857, 914, 915, 916, 918, 919, 926, 1107, 1117, 1119, 1120, 1122, 1123, 1125, 1128, 1130, 1131, 1132]
            # switch_time = [745,1110]
        elif self.dataSet == "nyse_n":
            switch_time_auto = [20, 21, 54, 101, 161, 171, 174, 187, 197, 199, 278, 279, 289, 340, 351, 365, 366, 367, 368, 371, 372, 413, 416, 440, 451, 455, 789, 792, 932, 933, 1010, 1011, 1651, 1652, 2051, 2052, 2351, 2353, 2769, 2770, 2901, 2902, 3401, 3404, 3501, 3508, 3551, 3562, 3833, 3834, 3851, 3901, 4214, 4231, 4251, 4252, 4301, 4333, 4589, 4595, 5120, 5151, 5396, 5398, 5454, 5456, 5540, 5541, 5551, 5601, 5661, 5677, 5699, 5701, 5743, 5801, 5803, 5804, 5851, 5993, 6046, 6051, 6052, 6095, 6097, 6103, 6155, 6201, 6202, 6251, 6254]

            switch_time = [5935,6270]



        # for i in switch_time:
        #     plt.axvline(x=i, ls="-", c="green")
        #
        # plt.axvline(x=20, ls="-", c="dimgray")
        # for j in switch_time_auto[0:]:
        #     plt.axvline(x=j, ls="-", c="black", linewidth=0.8)

        # plt.axhline(y=portfolio["SWITCH"].iloc[-1], ls="dashed", c="black", linewidth=0.6)
        # plt.text(x=portfolio["S4"].__len__()-1, y=portfolio["S4"].iloc[-1], s='%.4f' % portfolio["S4"].iloc[-1], ha='center', va='bottom')
        algos =  ['PPT',"OLMAR","BCRP","RMR"]
        # algos = ['MAXUP_T', 'MINUP_T', 'MAXDOWN_T', 'MINDOWN_T','MAXUP_F', 'MINUP_F', 'MAXDOWN_F', 'MINDOWN_F']
        for algo in algos:
            plt.text(x=portfolio[algo].__len__()-1, y=portfolio[algo].iloc[-1], s='%.4f' % portfolio[algo].iloc[-1], ha='center', va='bottom')
        plt.legend()
        plt.show()
        # fig1.savefig(self.fileName, format='eps')
#
