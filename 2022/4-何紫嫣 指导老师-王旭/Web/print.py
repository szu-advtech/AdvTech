import os
import sys
import time
from shutil import copyfile
# time.sleep(5)
arg = sys.argv[1]

dir = "/home/hzy/dataset/openvslam_erp/" + arg
print(dir)
# f = open(dir+'/x.txt', 'w+')
# f = open("/home/hzy/code/potree/potree/upload/x.txt", "w+")
# f.write(arg)
# f.close()
copyfile("/home/code/web/getresult.js", "/home/code/web/upload/getresult.js")