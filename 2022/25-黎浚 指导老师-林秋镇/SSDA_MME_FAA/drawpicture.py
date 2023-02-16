import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.interpolate import interp1d
MME_RDAtestroot="MME_RDA_test_loss.csv"
MME_RDAvalroot="MME_RDA_val_loss.csv"
MME_testroot="MME_test_loss.csv"
MME_valroot="MME_val_loss.csv"

# plot double lines
def plot_double_lines(n, x, y1, y2, pic_name):
    # initialize plot parameters
    print('picture name: %s, len of data: %d' % (pic_name, n))
    plt.rcParams['figure.figsize'] = (10 * 16 / 9, 10)
    plt.subplots_adjust(left=0.06, right=0.94, top=0.92, bottom=0.08)

    # 对x和y1进行插值
    x_smooth = np.linspace(x.min(), x.max(), 150)
    y1_smooth = make_interp_spline(x, y1)(x_smooth)
    # plot curve 1
    plt.plot(x_smooth, y1_smooth, label='MME')

    # 对x和y2进行插值
    x_smooth = np.linspace(x.min(), x.max(), 150)
    y2_smooth = make_interp_spline(x, y2)(x_smooth)
    # plot curve 2
    plt.plot(x_smooth, y2_smooth, label='MME_FAA')

    # show the legend
    plt.legend()

    # show the picture
    plt.savefig("test2.png")

if __name__ == '__main__':
    x=np.array(pd.read_csv(MME_valroot,header=0).iloc[0:40,1])
    y1s = np.array(pd.read_csv(MME_valroot,header=0).iloc[0:40,2])
    y2s = np.array(pd.read_csv(MME_RDAvalroot,header=0).iloc[0:40,2])
    plot_double_lines(len(x), x, y1s, y2s, 'Visualization of Linking Prediction')