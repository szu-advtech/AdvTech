from matplotlib import pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator, FuncFormatter


def to_percent(temp, position):
    return '%1.0f'%(100*temp) + '%'

if __name__ == '__main__':
    df=pd.read_csv('./EVM_of_letter.csv', header=None)
    df1=pd.read_csv('./others_of_lettter.csv', header=None)
    Y=df.iloc[:,0]
    X=df.iloc[:,1]
    w_svm=df1.iloc[:,0]
    rest_svm=df1.iloc[:,1]
    NNCAP=df1.iloc[:,2]
    # y2=[0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9,0.9]
    x_major_locator = MultipleLocator(0.1)
    # 把x轴的刻度间隔设置为1，并存在变量里
    y_major_locator = MultipleLocator(0.02)
    ax = plt.gca()
    # ax为两条坐标轴的实例
    ax.xaxis.set_major_locator(x_major_locator)
    # 把x轴的主刻度设置为1的倍数
    ax.yaxis.set_major_locator(y_major_locator)
    # 把y轴的主刻度设置为10的倍数
    plt.xlim(0, 0.45)
    # 把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
    plt.ylim(0.74, 1)
    # 把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
    plt.gca().xaxis.set_major_formatter(FuncFormatter(to_percent))
    plt.rcParams["font.sans-serif"] = "SimHei"
    plt.rcParams["axes.unicode_minus"] = False
    plt.plot(X, Y, color='b',linestyle='--',label='EVM')
    plt.plot(X, w_svm, color='r',linestyle='--',label='W-SVM')
    plt.plot(X, rest_svm, color='g',label='1-v-Rest-SVM+platt')
    plt.plot(X, NNCAP, color='#800080',label='NN+CAP')
    plt.xlabel('% of Unknown classes',fontsize=14)
    # y轴文本
    plt.ylabel('Open Set Recognition F1-Measure',fontsize=14)
    plt.legend(loc='upper right', ncol=1, fancybox=True, shadow=True)
    # 标题
    plt.title('letter数据集实验')
    plt.savefig('../images/equence.jpg', dpi=600, bbox_inches='tight')
    #第3步：显示图形
    plt.show()