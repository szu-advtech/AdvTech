
from MyTools import MyTools
import os
import datetime as dt

class MyLogger:

    def _createShareLog(self, filename):

        self.fileNoShare = 0
        self.pathShare = MyTools.getLogDir() + filename + '_log_'
        self.countShare = 0
        self.pid = os.getpid()
        # don't clear history of shared log
        # oldfileShare = Path(self.pathShare + str(self.fileNoShare))
        # if oldfileShare.exists():
        #     open(self.pathShare + str(self.fileNoShare), 'w').close()

        self.loggerShare = open(self.pathShare + str(self.fileNoShare), 'a+')

    def _createProcessLog(self, filename):

        self.fileNo = 0

        self.path = MyTools.getLogDir() + filename + '.log.'
        self.pid = os.getpid()
        self.path += str(self.pid) + '.'        # with processId

        self.count = 0
        self.logger = open(self.path + str(self.fileNo), 'a+')

    def __init__(self, filename, createProcessLog=True):
        """
        Creates a rotating log
        """
        self._createShareLog(filename)
        self.logger = None
        if createProcessLog:
            self._createProcessLog(filename)

        self.isWritingFinished= True
        self.isWritingFinishedShare = True

        self.indents = {}
        ind = ''
        self.indents[0] = ind
        for i in range(1, 20, 1):
            ind += '\t'
            self.indents[i] = ind

    def write(self, info, indent=0):

        if not self.logger:
            print('self.logger is None, write failed')
            return

        if self.isWritingFinished:
            self.isWritingFinished = False

            timeNow = dt.datetime.now()
            try:
                tmp = self.indents[indent] + str(self.pid) + ' ' + str(timeNow) + ' ' + info + '\n'
                self.logger.write(tmp)
                self.logger.flush()
            except Exception as e:
                pass

            self.count += 1
            if self.count >= 6000:

                # open a new log file
                self.count = 0
                self.fileNo += 1
                self.fileNo %= 5            # write a new file
                # clear it if it exists
                # open(self.path + str(self.fileNo), 'w').close()
                # reopen it for appending
                # close cur log
                self.loggerShare.write('..... reopen: ' + self.path + str(self.fileNo) + '\n')

                self.logger.close()
                open(self.path + str(self.fileNo), 'w').close()
                self.logger = open(self.path + str(self.fileNo), 'a+')
                self.logger.truncate()

            self.isWritingFinished = True

    def writeShare(self, info, indent=0):

        timeNow = dt.datetime.now()
        if self.isWritingFinishedShare:
            self.isWritingFinishedShare = False

            try:
                indent %= len(self.indents)
                tmp = self.indents[indent] + str(self.pid) + ' ' + str(timeNow) + ' ' + info + '\n'
                self.loggerShare.write(tmp)
                self.loggerShare.flush()
            except Exception as e:
                pass

            self.countShare += 1
            if self.countShare >= 6000:

                # close cur log
                self.loggerShare.close()

                # open a new log file
                self.countShare = 0
                self.fileNoShare += 1
                self.fileNoShare %= 5            # write a new file
                # clear it if it exists
                # open(self.path + str(self.fileNo), 'w').close()
                # reopen it for appending
                # print('..... reopen: ', self.path + str(self.fileNo))
                open(self.pathShare + str(self.fileNoShare), 'w').close()
                self.loggerShare = open(self.pathShare + str(self.fileNoShare), 'a+')
                self.loggerShare.truncate()

            self.isWritingFinishedShare = True

# ----------------------------------------------------------------------
if __name__ == "__main__":
    log_file = "test"
    mg = MyLogger(log_file)
    mg.write('sdfasdfsf')