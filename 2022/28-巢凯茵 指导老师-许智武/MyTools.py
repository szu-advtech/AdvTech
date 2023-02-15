
from decimal import Decimal
import json
import sys
import subprocess, signal, os


class MyTools:

    @staticmethod
    def getRound(floatNum, precision=4):
        return float(round(Decimal(floatNum), precision))
    @staticmethod
    def checkEqual(pred, actual, msg):
        diff = abs(pred - actual)
        assert diff < 1e-8, \
            "pred=" + str(pred) + ", actal=" + str(actual) + ' : ' + msg

    @staticmethod
    def checkLessEqualThan(pred, actual, msg):
        assert pred <= actual, \
            "pred=" + str(pred) + ", actal=" + str(actual) + ' : ' + msg

    @staticmethod
    def checkBigEqualThan(pred, actual, msg):
        assert pred >= actual, \
            "pred=" + str(pred) + ", actal=" + str(actual) + ' : ' + msg

    @staticmethod
    def insertSortedTupleList(a, tup, lo=0, hi=None):
        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid][0] < tup[0]:
                lo = mid + 1
            else:
                hi = mid
        a.insert(lo, tup)

    @staticmethod
    def insertSortedList(a, ele, lo=0, hi=None):
        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        while lo < hi:
            mid = (lo + hi) // 2
            if a[mid] < ele:
                lo = mid + 1
            else:
                hi = mid
        a.insert(lo, ele)

    @staticmethod
    def getDicByJson(astr, debug=True, info=''):
        # if debug:
        #     print('\n' + info + '-: ' + str(type(astr)) + '....', astr)
        if type(astr) is dict:
            return astr

        astr = str(astr)
        astr = astr.replace("\'", """"\"""")
        # if debug:
        #     print('\n' + info + '....tostr...', astr)
        dic = json.loads(astr)
        # if debug:
        #     print('\n' + info + '....after loads...', dic)
        return dic

    @staticmethod
    def getLogDir():
        startPath = sys.argv[0]
        # print('arg0: ', startPath)
        endIndex = startPath.rfind('\\')
        srcDir = startPath[0:endIndex]
        srcDir += '\\log\\'
        # print(srcDir)
        return srcDir

    @staticmethod
    def getStartDir():
        startPath = sys.argv[0]
        endIndex = startPath.rfind('/')
        srcDir = startPath[0:endIndex]
        return srcDir + '/'

    @staticmethod
    def killProcess(pidStr):
        p = subprocess.Popen(['ps', '-ef'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            line = line.decode("utf-8")
            line = line.lower()
            lst = line.split()
            if lst[1] == pidStr:
                pid = int(lst[1])
                print('killProcess......kill...pid=', pid)
                try:
                    os.kill(pid, signal.SIGKILL)
                except Exception as e:
                    print('----killProcess----------> ', e)

            # if '?' in line \
            #    and 'jianfei' in line \
            #    and 'python3' in line \
            #    and 'startOld.py' in line:
            #     lst = line.split()
            #     # print(str(lst[1]))
            #     pid = int(lst[1])
            #     os.kill(pid, signal.SIGKILL)
            #     continue

    @staticmethod
    def killProcessIfNotPid(pidStr, key1):
        p = subprocess.Popen(['ps', '-ef'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        for line in out.splitlines():
            line = line.decode("utf-8")
            line = line.lower()
            if key1 in line:
                lst = line.split()
                if lst[1] != pidStr:
                    pid = int(lst[1])
                    print('killProcessIfNotPid......kill...pid=', pid)
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except Exception as e:
                        print('----killProcessIfNotPid----------> ', e)

    # @staticmethod
    # def killProcessByciKey(caseInsensivekey):
    #     p = subprocess.Popen(['ps', '-ef'], stdout=subprocess.PIPE)
    #     out, err = p.communicate()
    #     for line in out.splitlines():
    #         line = line.decode("utf-8")
    #         line = line.lower()
    #         key = caseInsensivekey.lower()
    #         if key in line:
    #             lst = line.split()
    #             pid = int(lst[1])
    #             print('killProcessByciKey......kill...pid=', pid, 'key= ', caseInsensivekey)
    #             try:
    #                 os.kill(pid, signal.SIGKILL)
    #             except Exception as e:
    #                 print('----killProcessByciKey----------> ', e)

    @staticmethod
    def getPids(caseInsensivekey):
        p = subprocess.Popen(['ps', '-ef'], stdout=subprocess.PIPE)
        out, err = p.communicate()
        pids = []
        for line in out.splitlines():
            line = line.decode("utf-8")
            line = line.lower()
            key = caseInsensivekey.lower()
            if key in line:
                lst = line.split()
                pid = int(lst[1])
                pids.append(pid)
        return pids

    @staticmethod
    def killProcessByPids(pids):
        for p in pids:
            try:
                os.kill(p, signal.SIGKILL)
            except Exception as e:
                print('----killProcessByPids..pid=', p, '..failed: ', e)


    @staticmethod
    def updateJsonStore(dic, fileName):
        path = MyTools.getStartDir() + 'store/'
        jfile = None
        try:
            open(path + fileName, 'w').close()      # clear content
            jfile = open(path + fileName, 'a+')
        except Exception as e:
            print('>>>>> updateJsonStore...failed for', fileName, '..', e)
            return

        r = json.dumps(dic)
        jfile.write(r)
        jfile.flush()
        jfile.close()

    @staticmethod
    def getJsonStore(fileName):
        path = MyTools.getStartDir() + 'store/'
        jfile = None
        try:
            jfile = open(path + fileName, 'r')
        except Exception as e:
            print('>>>>> getJsonStore...failed for', fileName, '..', e)
            return None

        dic = json.load(jfile)
        dic = MyTools.getDicByJson(dic)
        return dic