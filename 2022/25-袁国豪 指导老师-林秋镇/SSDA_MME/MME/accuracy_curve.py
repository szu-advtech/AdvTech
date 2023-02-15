import matplotlib.pyplot as plt
import numpy as np
y_old_acc = [54.191176,54.191176,54.191176,54.852941,54.852941,54.852941,57.132353,57.132353,57.132353,57.132353,57.132353,59.852941,59.852941,60.551471]
y_new_acc = [56.470588,56.470588,56.875000,56.838235,56.838235,56.838235,56.948529,56.948529,57.720588,57.720588,58.014706,58.676471,58.676471,59.080882]
x_test_acc = [500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000]  # loss的数量，即x轴
"""
plt.figure()

# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epochs')  # x轴标签
plt.ylabel('accuracy')  # y轴标签

# 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 增加参数color='red',这是红色。
plt.plot(x_test_acc, y_old_acc, linewidth=1,color='red', linestyle="solid", label="lod loss")
plt.plot(x_test_acc, y_new_acc, linewidth=1,color='blue', linestyle="solid", label="new loss")
plt.legend()
plt.title('Accuracy curve')
plt.show()
"""


# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)

new_train_acc_path = "./new_train_acc.txt"  # 存储文件路径
old_train_acc_path = "./train_acc.txt"  # 存储文件路径

new_y_train_acc = data_read(new_train_acc_path)  # 训练准确率值，即y轴
new_x_train_acc = range(len(new_y_train_acc))  # 训练阶段准确率的数量，即x轴

old_y_train_acc = data_read(old_train_acc_path)  # 训练准确率值，即y轴
old_x_train_acc = range(len(old_y_train_acc))  # 训练阶段准确率的数量，即x轴

plt.figure()

# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('epochs')  # x轴标签
plt.ylabel('accuracy')  # y轴标签

# 以x_train_acc为横坐标，y_train_acc为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 增加参数color='red',这是红色。
plt.plot(new_x_train_acc, new_y_train_acc, color='red', linewidth=1, linestyle="solid", label="new train acc")
plt.plot(old_x_train_acc, old_y_train_acc, color='black', linewidth=1, linestyle="solid", label="old train acc")
plt.legend()
plt.title('Accuracy curve')
plt.show()

