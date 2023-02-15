import numpy as np
from matplotlib import pyplot as plt
#from utils import *


def data_read():               # 读取存储为txt文件的数据
    with open('C:\\Users\\James\\Desktop\\trn_loss.txt', "r") as f:
        #line = f.readline()    # 读取一行
        lines = f.readlines()    #读取整个文件所有行，保存在 list 列表中
        
    for i in range(len(lines)):
        lines[i] = lines[i].strip('\n')
        lines[i] = lines[i][1:-1]
        #raw_data = f.read()           # 读取一个列表
        #data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(lines, float)   # str->float


if __name__ == "__main__":

    y_train_loss = data_read()        # loss值，即y轴
    x_train_loss = range(len(y_train_loss))    # loss的数量，即x轴

    plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')    # x轴标签
    plt.ylabel('loss')     # y轴标签
	
	# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
	# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle="solid", label="train loss")
    plt.legend()
    plt.title('Loss curve')
    plt.show()


"""
#<https://huaweicloud.csdn.net/63802eecdacf622b8df86252.html?spm=1001.2101.3001.6650.3&utm_medium=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~activity-3-125661871-blog-106490754.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2~default~CTRLIST~activity-3-125661871-blog-106490754.pc_relevant_default&utm_relevant_index=5>
if __name__ == "__main__":

	train_acc_path = r"E:\relate_code\Gaitpart-master\train_acc.txt"   # 存储文件路径
	
	y_train_acc = data_read(train_acc_path)       # 训练准确率值，即y轴
	x_train_acc = range(len(y_train_acc))			 # 训练阶段准确率的数量，即x轴

	plt.figure()

    # 去除顶部和右边框框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epochs')    # x轴标签
    plt.ylabel('accuracy')     # y轴标签
	
	# 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
	# 增加参数color='red',这是红色。
    plt.plot(x_train_acc, y_train_acc, color='red',linewidth=1, linestyle="solid", label="train acc")
    plt.legend()
    plt.title('Accuracy curve')
    plt.show()

"""