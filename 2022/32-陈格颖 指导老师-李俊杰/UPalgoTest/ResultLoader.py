

from universal.result import AlgoResult
from MultiShower import MultiShower
import matplotlib
import matplotlib.pyplot as plt
import datetime
from MyLogger import MyLogger

class ResultLoader:

    def __init__(self):
        self.tMark = str(datetime.datetime.now())
        self.logger = MyLogger('Damp_')

    def getSavePath(self, databaseName, N):
        return str(databaseName) + '_' + str(N) + '_' + self.tMark + '_'

    def loadTest_DJIA(self):

        path = '/home/aze/project/UPalgoTest/resultSave/'
        # path = '/home/linuxbrew/XPshared/UPalgoTest/resultSave/'

        # savePath = self.getSavePath('DJIA30', 30)

        anti = AlgoResult.load(path + 'djia_30_2020-11-30 16:29:38.892668_ANTICOR')
        bah =  AlgoResult.load(path + 'djia_30_2020-11-30 16:29:38.892668_BAH')
        bcrp = AlgoResult.load(path + 'djia_30_2020-11-30 16:29:38.892668_BCRP')
        # eg = AlgoResult.load(path + 'djia_30_2020-03-21 04:17:49.190626_EG')
        olmar = AlgoResult.load(path + 'djia_30_2020-11-30 16:29:38.892668_OLMAR')
        rss_tr = AlgoResult.load(path + 'djia_30_2021-01-29 15:45:49.448141_OLMAR_RSS_BAH')
        # rss_tr = AlgoResult.load(path + 'djia_30_2020-11-22 12:07:18.909270_OLMAR_RSS')
        craps = AlgoResult.load(path + 'djia_30_2020-11-29 12:01:30.326022_CRAPS')
        raps = AlgoResult.load(path + 'djia_30_2020-11-29 11:25:47.473302_RAPS')
        lsrt = AlgoResult.load(path + 'djia_30_2020-11-29 11:25:47.473302_LSRT')
        acss = AlgoResult.load(path + 'djia_30_2020-11-30 15:59:23.383971_ACSS')
        pswd = AlgoResult.load(path + 'djia_30_2020-11-29 20:45:04.992528_PSWD')
        spolc = AlgoResult.load(path + 'djia_30_2021-01-24 16:22:33.014877_SPOLC')
        # ons = AlgoResult.load(path + 'djia_30_2020-03-21 04:17:49.190626_ONS')
        # pamr = AlgoResult.load(path + 'djia_30_2020-03-21 04:17:49.190626_PAMR')
        # wfm = AlgoResult.load(path + 'djia_30_2020-03-23 00:25:58.417441_WFM')
        # wfm = AlgoResult.load(path + 'djia_30_2020-03-23 22:26:51.741457_WFM')
        # wfm = AlgoResult.load(path + 'djia_30_2020-03-22 22:38:30.296127_WFM')
        # wfm = AlgoResult.load(path + 'djia_30_2020-03-23 00:25:58.417441_WFM')
        # wfm = AlgoResult.load(path + 'djia_30_2020-03-24 18:25:48.844277_WFM')
        save = 'DJIA30_T'
        ms = MultiShower('DJIA_30' + '_Result_')
        # result_up     = UP().run(self.data)
        fee = 0.001

        # ms.fileName = save + str(fee)

        ###########################
        craps.fee = fee
        raps.fee = fee
        lsrt.fee = fee
        acss.fee = fee
        anti.fee = fee
        bah.fee = fee
        bcrp.fee = fee
        # eg.fee = fee
        olmar.fee = fee
        rss_tr.fee = fee
        pswd.fee = fee
        spolc.fee = fee
        # ons.fee = fee
        # pamr.fee = fee
        # wfm.fee = fee
        self.logger.write('DJIA30' + '_Anticor_' + str(fee) + '_' + anti.summary())
        self.logger.write('DJIA30' + '_BAH_' + str(fee) + '_' + bah.summary())
        self.logger.write('DJIA30' + '_BCRP_' + str(fee) + '_' + bcrp.summary())
        # self.logger.write('DJIA30' + '_EG_' + str(fee) + '_' + eg.summary())
        self.logger.write('DJIA30' + '_OLMAR_' + str(fee) + '_' + olmar.summary())
        self.logger.write('DJIA30' + '_RSStr_' + str(fee) + '_' + rss_tr.summary())
        self.logger.write('DJIA30' + '_CRAPS_' + str(fee) + '_' + craps.summary())
        self.logger.write('DJIA30' + '_RAPS_' + str(fee) + '_' + raps.summary())
        self.logger.write('DJIA30' + '_LSRT_' + str(fee) + '_' + lsrt.summary())
        self.logger.write('DJIA30' + '_ACSS_' + str(fee) + '_' + acss.summary())
        self.logger.write('DJIA30' + '_PSWD_' + str(fee) + '_' + pswd.summary())
        self.logger.write('DJIA30' + '_SPOLC_' + str(fee) + '_' + spolc.summary())
        # self.logger.write('DJIA30' + '_ONS_' + str(fee) + '_' + ons.summary())
        # self.logger.write('DJIA30' + '_PAMR_' + str(fee) + '_' + pamr.summary())
        # self.logger.write('DJIA30' + '_WFM_' + str(fee) + '_' + wfm.summary())

        ms.show([anti, bah, bcrp,
                 olmar, rss_tr, craps,
                 raps, lsrt, acss, pswd, spolc],
                     ['ANTICOR', 'BAH', 'BCRP',
                      'OLMAR', 'PROC', 'CRAPS',
                      'RAPS', 'LSRT', 'ACSS', 'PSWD', 'SPOLC'])

        plt.show()

    def loadTest_MSCI(self):

        path = '/home/aze/project/UPalgoTest/resultSave/'

        # savePath = self.getSavePath('DJIA30', 30)

        anti = AlgoResult.load(path + 'msci_24_2020-12-01 17:17:20.712438_ANTICOR')
        bah = AlgoResult.load(path + 'msci_24_2020-12-01 17:17:20.712438_BAH')
        bcrp = AlgoResult.load(path + 'msci_24_2020-12-01 17:17:20.712438_BCRP')
        # eg = AlgoResult.load(path + 'msci_24_2020-03-21 04:40:21.535706_EG')
        olmar = AlgoResult.load(path + 'msci_24_2020-12-01 17:17:20.712438_OLMAR')
        # ons = AlgoResult.load(path + 'msci_24_2020-03-21 04:40:21.535706_ONS')
        # pamr = AlgoResult.load(path + 'msci_24_2020-03-21 04:40:21.535706_PAMR')
        # wfm = AlgoResult.load(path + 'msci_24_2020-03-21 05:20:20.686074_WFM')
        rss_tr = AlgoResult.load(path + 'msci_24_2021-01-29 15:46:37.035735_OLMAR_RSS_BAH')
        craps = AlgoResult.load(path + 'msci_24_2020-11-29 16:03:54.344904_CRAPS')
        raps = AlgoResult.load(path + 'msci_24_2020-11-29 16:03:54.344904_RAPS')
        lsrt = AlgoResult.load(path + 'msci_24_2020-11-29 16:03:54.344904_LSRT')
        acss = AlgoResult.load(path + 'msci_24_2020-11-29 16:03:54.344904_ACSS')
        spolc = AlgoResult.load(path + 'msci_24_2021-01-24 18:40:06.380750_SPOLC')


        save = 'MSCI24_T'
        ms = MultiShower('MSCI_24' + '_Result_')
        # result_up     = UP().run(self.data)
        fee = 0.001

        # ms.fileName = save + str(fee)

        ###########################
        craps.fee = fee
        raps.fee = fee
        lsrt.fee = fee
        acss.fee = fee
        anti.fee = fee
        bah.fee = fee
        bcrp.fee = fee
        # eg.fee = fee
        spolc.fee = fee
        olmar.fee = fee
        rss_tr.fee = fee
        self.logger.write('MSCI24' + '_Anticor_' + str(fee) + '_' + anti.summary())
        self.logger.write('MSCI24' + '_BAH_' + str(fee) + '_' + bah.summary())
        self.logger.write('MSCI24' + '_BCRP_' + str(fee) + '_' + bcrp.summary())
        # self.logger.write('MSCI24' + '_EG_' + str(fee) + '_' + eg.summary())
        self.logger.write('MSCI24' + '_OLMAR_' + str(fee) + '_' + olmar.summary())
        # self.logger.write('MSCI24' + '_ONS_' + str(fee) + '_' + ons.summary())
        # self.logger.write('MSCI24' + '_PAMR_' + str(fee) + '_' + pamr.summary())
        # self.logger.write('MSCI24' + '_WFM_' + str(fee) + '_' + wfm.summary())
        self.logger.write('MSCI24' + '_RSStr_' + str(fee) + '_' + rss_tr.summary())
        self.logger.write('MSCI24' + '_CRAPS_' + str(fee) + '_' + craps.summary())
        self.logger.write('MSCI24' + '_RAPS_' + str(fee) + '_' + raps.summary())
        self.logger.write('MSCI24' + '_LSRT_' + str(fee) + '_' + lsrt.summary())
        self.logger.write('MSCI24' + '_ACSS_' + str(fee) + '_' + acss.summary())
        self.logger.write('MSCI24' + '_SPOLC_' + str(fee) + '_' + spolc.summary())


        ms.show([anti, bah, bcrp,
                 olmar, rss_tr, craps,
                 raps, lsrt, acss, spolc],
                ['ANTICOR', 'BAH', 'BCRP',
                 'OLMAR', 'PROC', 'CRAPS',
                 'RAPS', 'LSRT', 'ACSS', 'SPOLC'])

        plt.show()


    def loadTest_NYSE_N(self):
        path = '/home/aze/project/UPalgoTest/resultSave/'

        # savePath = self.getSavePath('DJIA30', 30)

        anti = AlgoResult.load(path + 'nyse_n_23_2021-01-24 20:48:22.391390_ANTICOR')
        bah = AlgoResult.load(path + 'nyse_n_23_2021-01-24 20:48:22.391390_BAH')
        bcrp = AlgoResult.load(path + 'nyse_n_23_2021-01-24 20:48:22.391390_BCRP')
        # eg = AlgoResult.load(path + 'msci_24_2020-03-21 04:40:21.535706_EG')
        olmar = AlgoResult.load(path + 'nyse_n_23_2021-01-24 20:48:22.391390_OLMAR')
        # ons = AlgoResult.load(path + 'msci_24_2020-03-21 04:40:21.535706_ONS')
        # pamr = AlgoResult.load(path + 'msci_24_2020-03-21 04:40:21.535706_PAMR')
        # wfm = AlgoResult.load(path + 'msci_24_2020-03-21 05:20:20.686074_WFM')
        rss_tr = AlgoResult.load(path + 'nyse_n_23_2020-12-22 17_57_29.786629_OLMAR_RSS')
        craps = AlgoResult.load(path + 'nyse_n_23_2021-01-24 20:48:22.391390_CRAPS')
        raps = AlgoResult.load(path + 'nyse_n_23_2021-01-24 20:48:22.391390_RAPS')
        lsrt = AlgoResult.load(path + 'nyse_n_23_2021-01-24 20:48:22.391390_LSRT')
        # acss = AlgoResult.load(path + 'msci_24_2020-11-29 16:03:54.344904_ACSS')
        spolc = AlgoResult.load(path + 'nyse_n_23_2021-01-24 20:48:22.391390_SPOLC')

        save = 'NYSE_N_T'
        ms = MultiShower('NYSE_N' + '_Result_')
        # result_up     = UP().run(self.data)
        fee = 0.001

        # ms.fileName = save + str(fee)

        ###########################
        craps.fee = fee
        raps.fee = fee
        lsrt.fee = fee
        # acss.fee = fee
        anti.fee = fee
        bah.fee = fee
        bcrp.fee = fee
        # eg.fee = fee
        spolc.fee = fee
        olmar.fee = fee
        rss_tr.fee = fee
        self.logger.write('NYSE_N' + '_Anticor_' + str(fee) + '_' + anti.summary())
        self.logger.write('NYSE_N' + '_BAH_' + str(fee) + '_' + bah.summary())
        self.logger.write('NYSE_N' + '_BCRP_' + str(fee) + '_' + bcrp.summary())
        # self.logger.write('MSCI24' + '_EG_' + str(fee) + '_' + eg.summary())
        self.logger.write('NYSE_N' + '_OLMAR_' + str(fee) + '_' + olmar.summary())
        # self.logger.write('MSCI24' + '_ONS_' + str(fee) + '_' + ons.summary())
        # self.logger.write('MSCI24' + '_PAMR_' + str(fee) + '_' + pamr.summary())
        # self.logger.write('MSCI24' + '_WFM_' + str(fee) + '_' + wfm.summary())
        self.logger.write('NYSE_N' + '_RSStr_' + str(fee) + '_' + rss_tr.summary())
        self.logger.write('NYSE_N' + '_CRAPS_' + str(fee) + '_' + craps.summary())
        self.logger.write('NYSE_N' + '_RAPS_' + str(fee) + '_' + raps.summary())
        self.logger.write('NYSE_N' + '_LSRT_' + str(fee) + '_' + lsrt.summary())
        # self.logger.write('NYSE_N' + '_ACSS_' + str(fee) + '_' + acss.summary())
        self.logger.write('NYSE_N' + '_SPOLC_' + str(fee) + '_' + spolc.summary())

        ms.show([anti, bah, bcrp,
                 olmar, rss_tr, craps,
                 raps, lsrt, spolc],
                ['ANTICOR', 'BAH', 'BCRP',
                 'OLMAR', 'PROC', 'CRAPS',
                 'RAPS', 'LSRT', 'SPOLC'])

        plt.show()


    def loadTest_NYSE_O(self):

        path = '/home/linuxbrew/XPshared/UPalgoTest/resultSave_tooHigh/'

        # savePath = self.getSavePath('DJIA30', 30)

        anti = AlgoResult.load(path + 'nyse_o_36_2020-03-21 05:52:37.366878_ANTICOR')
        bah = AlgoResult.load(path + 'nyse_o_36_2020-03-21 05:52:37.366878_BAH')
        bcrp = AlgoResult.load(path + 'nyse_o_36_2020-03-21 05:52:37.366878_BCRP')
        eg = AlgoResult.load(path + 'nyse_o_36_2020-03-21 05:52:37.366878_EG')
        olmar = AlgoResult.load(path + 'nyse_o_36_2020-03-21 05:52:37.366878_OLMAR')
        ons = AlgoResult.load(path + 'nyse_o_36_2020-03-21 05:52:37.366878_ONS')
        pamr = AlgoResult.load(path + 'nyse_o_36_2020-03-21 05:52:37.366878_PAMR')
        wfm = AlgoResult.load(path + 'nyse_o_36_2020-03-21 05:52:37.366878_WFM')
        # wfm = AlgoResult.load(path + 'nyse_o_36_2020-03-24 15:41:11.609193_WFM')
        # wfm = AlgoResult.load(path + 'nyse_o_36_2020-03-24 10:15:25.541041_WFM')
        # wfm = AlgoResult.load(path + 'nyse_o_36_2020-03-24 10:15:25.541041_WFM')
        save = 'NYSE_O_T'
        ms = MultiShower('NYSE_O_36' + '_Result_')
        # result_up     = UP().run(self.data)
        for fee in [0, 0.001, 0.003, 0.005, 0.008, 0.01]:
            ms.fileName = save + str(fee)

            ###########################
            anti.fee = fee
            bah.fee = fee
            bcrp.fee = fee
            eg.fee = fee
            olmar.fee = fee
            ons.fee = fee
            pamr.fee = fee
            wfm.fee = fee
            self.logger.write('NYSE_O_36' + '_Anticor_' + str(fee) + '_' + anti.summary())
            self.logger.write('NYSE_O_36' + '_BAH_' + str(fee) + '_' + bah.summary())
            self.logger.write('NYSE_O_36' + '_BCRP_' + str(fee) + '_' + bcrp.summary())
            self.logger.write('NYSE_O_36' + '_EG_' + str(fee) + '_' + eg.summary())
            self.logger.write('NYSE_O_36' + '_OLMAR_' + str(fee) + '_' + olmar.summary())
            self.logger.write('NYSE_O_36' + '_ONS_' + str(fee) + '_' + ons.summary())
            self.logger.write('NYSE_O_36' + '_PAMR_' + str(fee) + '_' + pamr.summary())
            self.logger.write('NYSE_O_36' + '_WFM_' + str(fee) + '_' + wfm.summary())

            ms.show([anti, bah,bcrp,
                     eg, olmar, ons, pamr, wfm
                     ],
                    ['ANTICOR', 'BAH', 'BCRP', 'EG',
                     'OLMAR', 'ONS', 'PAMR', 'WFM'])

        plt.show()

    def loadTest_SP500_O(self):

        path = '/home/aze/project/UPalgoTest/resultSave/'

        # savePath = self.getSavePath('DJIA30', 30)

        anti = AlgoResult.load(path + 'sp500_25_2020-12-01 17:30:56.480181_ANTICOR')
        bah = AlgoResult.load(path + 'sp500_25_2020-12-01 17:30:56.480181_BAH')
        bcrp = AlgoResult.load(path + 'sp500_25_2020-12-01 17:30:56.480181_BCRP')
        # eg = AlgoResult.load(path + 'sp500_25_2020-03-21 04:27:37.704643_EG')
        olmar = AlgoResult.load(path + 'sp500_25_2020-12-01 17:30:56.480181_OLMAR')
        # ons = AlgoResult.load(path + 'sp500_25_2020-03-21 04:27:37.704643_ONS')
        # pamr = AlgoResult.load(path + 'sp500_25_2020-03-21 04:27:37.704643_PAMR')
        # wfm = AlgoResult.load(path + 'sp500_25_2020-03-21 07:25:37.195994_WFM')
        # wfm = AlgoResult.load(path + 'sp500_25_2020-03-24 15:58:11.126315_WFM')
        # wfm = AlgoResult.load(path + 'sp500_25_2020-03-24 20:04:06.347907_WFM')
        rss_tr = AlgoResult.load(path + 'sp500_25_2020-12-29 15:55:51.962811_OLMAR_RSS_BAH')
        craps = AlgoResult.load(path + 'sp500_25_2020-11-29 12:58:41.408522_CRAPS')
        raps = AlgoResult.load(path + 'sp500_25_2020-11-29 12:58:41.408522_RAPS')
        lsrt = AlgoResult.load(path + 'sp500_25_2020-11-29 12:58:41.408522_LSRT')
        acss = AlgoResult.load(path + 'sp500_25_2020-11-30 16:01:50.509573_ACSS')
        pswd = AlgoResult.load(path + 'sp500_25_2020-11-29 20:49:14.225289_PSWD')
        spolc = AlgoResult.load(path + 'sp500_25_2021-01-25 14:48:08.994297_SPOLC')
        save = 'SP500_T'
        ms = MultiShower('SP500_T_25' + '_Result_')
        # result_up     = UP().run(self.data)
        fee = 0.001

        # ms.fileName = save + str(fee)

        ###########################
        craps.fee = fee
        raps.fee = fee
        lsrt.fee = fee
        acss.fee = fee
        anti.fee = fee
        bah.fee = fee
        bcrp.fee = fee
        # eg.fee = fee
        olmar.fee = fee
        rss_tr.fee = fee
        pswd.fee = fee
        spolc.fee = fee
        self.logger.write('SP500_T_25' + '_Anticor_' + str(fee) + '_' + anti.summary())
        self.logger.write('SP500_T_25' + '_BAH_' + str(fee) + '_' + bah.summary())
        self.logger.write('SP500_T_25' + '_BCRP_' + str(fee) + '_' + bcrp.summary())
        # self.logger.write('MSCI24' + '_EG_' + str(fee) + '_' + eg.summary())
        self.logger.write('SP500_T_25' + '_OLMAR_' + str(fee) + '_' + olmar.summary())
        # self.logger.write('MSCI24' + '_ONS_' + str(fee) + '_' + ons.summary())
        # self.logger.write('MSCI24' + '_PAMR_' + str(fee) + '_' + pamr.summary())
        # self.logger.write('MSCI24' + '_WFM_' + str(fee) + '_' + wfm.summary())
        self.logger.write('SP500_T_25' + '_RSStr_' + str(fee) + '_' + rss_tr.summary())
        self.logger.write('SP500_T_25' + '_CRAPS_' + str(fee) + '_' + craps.summary())
        self.logger.write('SP500_T_25' + '_RAPS_' + str(fee) + '_' + raps.summary())
        self.logger.write('SP500_T_25' + '_LSRT_' + str(fee) + '_' + lsrt.summary())
        self.logger.write('SP500_T_25' + '_ACSS_' + str(fee) + '_' + acss.summary())
        # self.logger.write('SP500_T_25' + '_PSWD_' + str(fee) + '_' + pswd.summary())
        self.logger.write('SP500_T_25' + '_SPOLC_' + str(fee) + '_' + spolc.summary())
        ms.show([anti, bah, bcrp,
                 olmar, rss_tr, craps,
                 raps, lsrt, acss, spolc],
                ['ANTICOR', 'BAH', 'BCRP',
                 'OLMAR', 'PROC', 'CRAPS',
                 'RAPS', 'LSRT', 'ACSS', 'SPOLC'])

        plt.show()

    def loadTest_TSE88(self):

        path = '/home/aze/project/UPalgoTest/resultSave/'

        # savePath = self.getSavePath('DJIA30', 30)

        anti = AlgoResult.load(path + 'tse_88_2020-12-01 17:43:40.379329_ANTICOR')
        bah = AlgoResult.load(path + 'tse_88_2020-12-01 17:43:40.379329_BAH')
        bcrp = AlgoResult.load(path + 'tse_88_2020-12-01 17:43:40.379329_BCRP')
        # eg = AlgoResult.load(path + 'tse_88_2020-03-21 04:36:55.338485_EG')
        olmar = AlgoResult.load(path + 'tse_88_2020-12-01 17:43:40.379329_OLMAR')
        # ons = AlgoResult.load(path + 'tse_88_2020-03-21 04:36:55.338485_ONS')
        # pamr = AlgoResult.load(path + 'tse_88_2020-03-21 04:36:55.338485_PAMR')
        # wfm = AlgoResult.load(path + 'tse_88_2020-03-22 22:00:23.506606_WFM')
        rss_tr = AlgoResult.load(path + 'tse_88_2020-12-16 09_32_57.597462_OLMAR_RSS')
        craps = AlgoResult.load(path + 'tse_88_2020-11-29 15:16:40.310326_CRAPS')
        raps = AlgoResult.load(path + 'tse_88_2020-11-29 15:16:40.310326_RAPS')
        lsrt = AlgoResult.load(path + 'tse_88_2020-11-29 15:16:40.310326_LSRT')
        acss = AlgoResult.load(path + 'tse_88_2020-11-30 16:01:41.094598_ACSS')
        # pswd = AlgoResult.load(path + 'tse_88_2020-12-01 18:04:56.415716_PSWD')
        spolc = AlgoResult.load(path + 'tse_88_2021-01-25 16:12:38.124735_SPOLC')

        save = 'TSE88_T'
        ms = MultiShower('TSE88_T_88' + '_Result_')
        # result_up     = UP().run(self.data)
        fee = 0.001

        # ms.fileName = save + str(fee)

        ###########################
        craps.fee = fee
        raps.fee = fee
        lsrt.fee = fee
        acss.fee = fee
        anti.fee = fee
        bah.fee = fee
        bcrp.fee = fee
        # eg.fee = fee
        olmar.fee = fee
        rss_tr.fee = fee
        # pswd.fee = fee
        spolc.fee = fee
        self.logger.write('TSE88_T_88' + '_Anticor_' + str(fee) + '_' + anti.summary())
        self.logger.write('TSE88_T_88' + '_BAH_' + str(fee) + '_' + bah.summary())
        self.logger.write('TSE88_T_88' + '_BCRP_' + str(fee) + '_' + bcrp.summary())
        # self.logger.write('MSCI24' + '_EG_' + str(fee) + '_' + eg.summary())
        self.logger.write('TSE88_T_88' + '_OLMAR_' + str(fee) + '_' + olmar.summary())
        # self.logger.write('MSCI24' + '_ONS_' + str(fee) + '_' + ons.summary())
        # self.logger.write('MSCI24' + '_PAMR_' + str(fee) + '_' + pamr.summary())
        # self.logger.write('MSCI24' + '_WFM_' + str(fee) + '_' + wfm.summary())
        self.logger.write('TSE88_T_88' + '_RSStr_' + str(fee) + '_' + rss_tr.summary())
        self.logger.write('TSE88_T_88' + '_CRAPS_' + str(fee) + '_' + craps.summary())
        self.logger.write('TSE88_T_88' + '_RAPS_' + str(fee) + '_' + raps.summary())
        self.logger.write('TSE88_T_88' + '_LSRT_' + str(fee) + '_' + lsrt.summary())
        self.logger.write('TSE88_T_88' + '_ACSS_' + str(fee) + '_' + acss.summary())
        self.logger.write('TSE88_T_88' + '_SPOLC_' + str(fee) + '_' + spolc.summary())
        # self.logger.write('TSE88_T_88' + '_PSWD_' + str(fee) + '_' + pswd.summary())


        ms.show([anti, bah, bcrp,
                 olmar, rss_tr, craps,
                 raps, lsrt, acss, spolc],
                ['ANTICOR', 'BAH', 'BCRP',
                 'OLMAR', 'PROC', 'CRAPS',
                 'RAPS', 'LSRT', 'ACSS', 'SPOLC'])

        plt.show()

    def loadTest_FTSE(self):

        path = '/home/aze/project/UPalgoTest/resultSave/'

        # savePath = self.getSavePath('DJIA30', 30)

        anti = AlgoResult.load(path + 'FTSE100_raw_83_2020-12-01 17:52:40.405016_ANTICOR')
        bah = AlgoResult.load(path + 'FTSE100_raw_83_2020-12-01 17:52:40.405016_BAH')
        bcrp = AlgoResult.load(path + 'FTSE100_raw_83_2020-12-01 17:52:40.405016_BCRP')
        # eg = AlgoResult.load(path + 'tse_88_2020-03-21 04:36:55.338485_EG')
        olmar = AlgoResult.load(path + 'FTSE100_raw_83_2020-12-01 17:52:40.405016_OLMAR')
        # ons = AlgoResult.load(path + 'tse_88_2020-03-21 04:36:55.338485_ONS')
        # pamr = AlgoResult.load(path + 'tse_88_2020-03-21 04:36:55.338485_PAMR')
        # wfm = AlgoResult.load(path + 'tse_88_2020-03-22 22:00:23.506606_WFM')
        rss_tr = AlgoResult.load(path + 'FTSE100_data_83_2020-11-27 14:20:29.716754_OLMAR_RSS')
        craps = AlgoResult.load(path + 'FTSE100_raw_83_2020-11-29 17:51:58.320106_CRAPS')
        raps = AlgoResult.load(path + 'FTSE100_raw_83_2020-11-29 17:51:58.320106_RAPS')
        lsrt = AlgoResult.load(path + 'FTSE100_raw_83_2020-11-29 17:51:58.320106_LSRT')
        acss = AlgoResult.load(path + 'FTSE100_raw_83_2020-11-29 17:51:58.320106_ACSS')

        save = 'FTSE83_T'
        ms = MultiShower('FTSE83_T' + '_Result_')
        # result_up     = UP().run(self.data)
        fee = 0.001

        # ms.fileName = save + str(fee)

        ###########################
        craps.fee = fee
        raps.fee = fee
        lsrt.fee = fee
        acss.fee = fee
        anti.fee = fee
        bah.fee = fee
        bcrp.fee = fee
        # eg.fee = fee
        olmar.fee = fee
        rss_tr.fee = fee
        self.logger.write('FTSE83_T' + '_Anticor_' + str(fee) + '_' + anti.summary())
        self.logger.write('FTSE83_T' + '_BAH_' + str(fee) + '_' + bah.summary())
        self.logger.write('FTSE83_T' + '_BCRP_' + str(fee) + '_' + bcrp.summary())
        # self.logger.write('MSCI24' + '_EG_' + str(fee) + '_' + eg.summary())
        self.logger.write('FTSE83_T' + '_OLMAR_' + str(fee) + '_' + olmar.summary())
        # self.logger.write('MSCI24' + '_ONS_' + str(fee) + '_' + ons.summary())
        # self.logger.write('MSCI24' + '_PAMR_' + str(fee) + '_' + pamr.summary())
        # self.logger.write('MSCI24' + '_WFM_' + str(fee) + '_' + wfm.summary())
        self.logger.write('FTSE83_T' + '_RSStr_' + str(fee) + '_' + rss_tr.summary())
        self.logger.write('FTSE83_T' + '_CRAPS_' + str(fee) + '_' + craps.summary())
        self.logger.write('FTSE83_T' + '_RAPS_' + str(fee) + '_' + raps.summary())
        self.logger.write('FTSE83_T' + '_LSRT_' + str(fee) + '_' + lsrt.summary())
        self.logger.write('FTSE83_T' + '_ACSS_' + str(fee) + '_' + acss.summary())

        ms.show([anti, bah, bcrp,
                 olmar, rss_tr, craps,
                 raps, lsrt, acss],
                ['ANTICOR', 'BAH', 'BCRP',
                 'OLMAR', 'RSStr', 'CRAPS',
                 'RAPS', 'LSRT', 'ACSS'])

        plt.show()

    def loadTest_STOXX(self):

        path = '/home/aze/project/UPalgoTest/resultSave/'

        # savePath = self.getSavePath('DJIA30', 30)

        anti = AlgoResult.load(path + 'STOXX50_data_49_2020-12-02 09:58:06.843640_ANTICOR')
        bah = AlgoResult.load(path + 'STOXX50_data_49_2020-12-02 09:58:06.843640_BAH')
        bcrp = AlgoResult.load(path + 'STOXX50_data_49_2020-12-02 09:58:06.843640_BCRP')
        # eg = AlgoResult.load(path + 'tse_88_2020-03-21 04:36:55.338485_EG')
        olmar = AlgoResult.load(path + 'STOXX50_data_49_2020-12-02 09:58:06.843640_OLMAR')
        # ons = AlgoResult.load(path + 'tse_88_2020-03-21 04:36:55.338485_ONS')
        # pamr = AlgoResult.load(path + 'tse_88_2020-03-21 04:36:55.338485_PAMR')
        # wfm = AlgoResult.load(path + 'tse_88_2020-03-22 22:00:23.506606_WFM')
        rss_tr = AlgoResult.load(path + 'STOXX50_data_49_2020-11-27 11:07:47.527673_OLMAR_RSS')
        craps = AlgoResult.load(path + 'STOXX50_data_49_2020-11-29 17:34:27.830751_CRAPS')
        raps = AlgoResult.load(path + 'STOXX50_data_49_2020-11-29 17:34:27.830751_RAPS')
        lsrt = AlgoResult.load(path + 'STOXX50_data_49_2020-11-29 17:34:27.830751_LSRT')
        acss = AlgoResult.load(path + 'STOXX50_data_49_2020-11-29 17:34:27.830751_ACSS')

        save = 'STOXX49_T'
        ms = MultiShower('STOXX49_T' + '_Result_')
        # result_up     = UP().run(self.data)
        fee = 0.001

        # ms.fileName = save + str(fee)

        ###########################
        craps.fee = fee
        raps.fee = fee
        lsrt.fee = fee
        acss.fee = fee
        anti.fee = fee
        bah.fee = fee
        bcrp.fee = fee
        # eg.fee = fee
        olmar.fee = fee
        rss_tr.fee = fee
        self.logger.write('STOXX50_T' + '_Anticor_' + str(fee) + '_' + anti.summary())
        self.logger.write('STOXX50_T' + '_BAH_' + str(fee) + '_' + bah.summary())
        self.logger.write('STOXX50_T' + '_BCRP_' + str(fee) + '_' + bcrp.summary())
        # self.logger.write('MSCI24' + '_EG_' + str(fee) + '_' + eg.summary())
        self.logger.write('STOXX50_T' + '_OLMAR_' + str(fee) + '_' + olmar.summary())
        # self.logger.write('MSCI24' + '_ONS_' + str(fee) + '_' + ons.summary())
        # self.logger.write('MSCI24' + '_PAMR_' + str(fee) + '_' + pamr.summary())
        # self.logger.write('MSCI24' + '_WFM_' + str(fee) + '_' + wfm.summary())
        self.logger.write('STOXX50_T' + '_RSStr_' + str(fee) + '_' + rss_tr.summary())
        self.logger.write('STOXX50_T' + '_CRAPS_' + str(fee) + '_' + craps.summary())
        self.logger.write('STOXX50_T' + '_RAPS_' + str(fee) + '_' + raps.summary())
        self.logger.write('STOXX50_T' + '_LSRT_' + str(fee) + '_' + lsrt.summary())
        self.logger.write('STOXX50_T' + '_ACSS_' + str(fee) + '_' + acss.summary())

        ms.show([anti, bah, bcrp,
                 olmar, rss_tr, craps,
                 raps, lsrt, acss],
                ['ANTICOR', 'BAH', 'BCRP',
                 'OLMAR', 'RSStr', 'CRAPS',
                 'RAPS', 'LSRT', 'ACSS'])

        plt.show()
    def loadTest_HS300(self):
        path = '/home/aze/project/UPalgoTest/resultSave/'

        # savePath = self.getSavePath('DJIA30', 30)

        anti = AlgoResult.load(path + 'hs300_44_2021-01-25 12:46:34.107784_ANTICOR')
        bah = AlgoResult.load(path + 'hs300_44_2021-01-25 12:46:34.107784_BAH')
        bcrp = AlgoResult.load(path + 'hs300_44_2021-01-25 12:46:34.107784_BCRP')
        # eg = AlgoResult.load(path + 'msci_24_2020-03-21 04:40:21.535706_EG')
        olmar = AlgoResult.load(path + 'hs300_44_2021-01-25 12:46:34.107784_OLMAR')
        # ons = AlgoResult.load(path + 'msci_24_2020-03-21 04:40:21.535706_ONS')
        # pamr = AlgoResult.load(path + 'msci_24_2020-03-21 04:40:21.535706_PAMR')
        # wfm = AlgoResult.load(path + 'msci_24_2020-03-21 05:20:20.686074_WFM')
        rss_tr = AlgoResult.load(path + 'hs300_44_2021-01-06 10:57:49.241085_OLMAR_RSS_BAH')
        craps = AlgoResult.load(path + 'hs300_44_2021-01-25 12:46:34.107784_CRAPS')
        raps = AlgoResult.load(path + 'hs300_44_2021-01-25 12:46:34.107784_RAPS')
        lsrt = AlgoResult.load(path + 'hs300_44_2021-01-25 12:46:34.107784_LSRT')
        acss = AlgoResult.load(path + 'hs300_44_2021-01-25 12:46:34.107784_ACSS')
        spolc = AlgoResult.load(path + 'hs300_44_2021-01-25 12:46:34.107784_SPOLC')

        save = 'HS300_T_'
        ms = MultiShower('HS300_T' + '_Result_')
        # result_up     = UP().run(self.data)
        fee = 0.001

        # ms.fileName = save + str(fee)

        ###########################
        craps.fee = fee
        raps.fee = fee
        lsrt.fee = fee
        acss.fee = fee
        anti.fee = fee
        bah.fee = fee
        bcrp.fee = fee
        # eg.fee = fee
        spolc.fee = fee
        olmar.fee = fee
        rss_tr.fee = fee
        self.logger.write('HS300' + '_Anticor_' + str(fee) + '_' + anti.summary())
        self.logger.write('HS300' + '_BAH_' + str(fee) + '_' + bah.summary())
        self.logger.write('HS300' + '_BCRP_' + str(fee) + '_' + bcrp.summary())
        # self.logger.write('MSCI24' + '_EG_' + str(fee) + '_' + eg.summary())
        self.logger.write('HS300' + '_OLMAR_' + str(fee) + '_' + olmar.summary())
        # # self.logger.write('MSCI24' + '_ONS_' + str(fee) + '_' + ons.summary())
        # # self.logger.write('MSCI24' + '_PAMR_' + str(fee) + '_' + pamr.summary())
        # # self.logger.write('MSCI24' + '_WFM_' + str(fee) + '_' + wfm.summary())
        self.logger.write('HS300' + '_RSStr_' + str(fee) + '_' + rss_tr.summary())
        self.logger.write('HS300' + '_CRAPS_' + str(fee) + '_' + craps.summary())
        self.logger.write('HS300' + '_RAPS_' + str(fee) + '_' + raps.summary())
        self.logger.write('HS300' + '_LSRT_' + str(fee) + '_' + lsrt.summary())
        self.logger.write('HS300' + '_ACSS_' + str(fee) + '_' + acss.summary())
        self.logger.write('HS300' + '_SPOLC_' + str(fee) + '_' + spolc.summary())

        # ms.show([spolc],['SPOLC'])
        ms.show([anti, bah, bcrp,
                 olmar, rss_tr, craps,
                 raps, lsrt, spolc, acss],
                ['ANTICOR', 'BAH', 'BCRP',
                 'OLMAR', 'PROC', 'CRAPS',
                 'RAPS', 'LSRT', 'SPOLC', 'ACSS'])

        plt.show()

    def plotSpecialParameter(self):
        path = '/home/aze/project/UPalgoTest/data/'
        djia = AlgoResult.load(path + 'djia_30_2021-03-17 15_26_42.918017_OLMAR_RSS_BAH')

        hs300 = AlgoResult.load(path + 'hs300_44_2021-03-17 16_24_34.789377_OLMAR_RSS_BAH')

        msci = AlgoResult.load(path + 'msci_24_2021-03-17 18_11_20.919663_OLMAR_RSS_BAH')

        nyse_n = AlgoResult.load(path + 'nyse_n_23_2021-03-17 19_06_26.952476_OLMAR_RSS_BAH')

        sp500 = AlgoResult.load(path + 'sp500_25_2021-03-17 17_12_50.623980_OLMAR_RSS_BAH')

        tse = AlgoResult.load(path + 'tse_88_2021-03-17 20_01_53.271202_OLMAR_RSS_BAH')


        ms = MultiShower('LrAndExpectaction' + '_Result_')
        self.logger.write('djia' + '_' + djia.summary())
        self.logger.write('msci' + '_' + msci.summary())
        self.logger.write('tse' + '_' + tse.summary())
        self.logger.write('sp500' + '_' + sp500.summary())
        self.logger.write('hs300' + '_' + hs300.summary())
        self.logger.write('nyse_n' + '_' + nyse_n.summary())

        ms.show([djia, msci, tse, sp500, hs300, nyse_n],
                ['DJIA', 'MSCI', 'TSE', 'SP500', 'HS300', 'NYSE_N'])

        plt.show()



if __name__ == '__main__':


    r = ResultLoader()
    # r.loadTest_DJIA()
    # r.loadTest_MSCI()
    # r.loadTest_NYSE_N()
    # r.loadTest_NYSE_O()
    # r.loadTest_SP500_O()
    r.plotSpecialParameter()