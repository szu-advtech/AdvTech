from matplotlib import pyplot as plt
import csv
import glob
from os.path import basename
from scipy.interpolate import make_interp_spline
import numpy as np

# fig = plt.figure(dpi=300, figsize=(10, 6))
plt.title("REINFORCE_jobs_burst", fontsize=16)
plt.xlabel("Iteration", fontsize=16)  # 横坐标
plt.ylabel("AVG_Return", fontsize=16)  # 纵坐标
files = sorted(glob.glob('avg_returns_reinforce_beta*.csv'))  # 读取目录下所有以.csv结尾的文件，这里可以添加绝对路径
for file in files:
    filename = basename(file).rsplit('.', 1)[
        0]  # 用.来分割文件名，取前半部分，例如XX.csv，取XX
    #     print('\r'+ filename + "  ", flush = True)
    with open(file) as f:
        csvreader = csv.reader(f, delimiter=",", quotechar='"')
        for line in range(1):  # 0代表从文件第一行开始读取
            next(csvreader)
        Wavelength = []
        Absorbance = []  # 横纵坐标分别建立了两个list
        for row in csvreader:
            Wavelength.append(float(row[0]))
            Absorbance.append(float(row[1]))  # 读取数据，放入list
        # 平滑处理
        Wavelength_array = np.array(Wavelength)
        Absorbance_array = np.array(Absorbance)  # list转array，为了调用下面的min,max
        plt.xlim((0, 20000))
        plt.ylim((-1000, 10000))
        x_smooth = np.linspace(Wavelength_array.min(), Wavelength_array.max(), 5)
        y_smooth = make_interp_spline(Wavelength_array, Absorbance_array)(x_smooth)
        # plt.plot(x_smooth, y_smooth)
        plt.plot(x_smooth, y_smooth, '-', label=filename)
        plt.legend()                                             #原本是做了一个右上角的label的，但是她不要

#       print("Done processing " + str(len(files)) + " files.")
plt.show()
# plt.savefig('图片名' + '.png', dpi = 300)   #不知道为什么保存了啥也看不到…