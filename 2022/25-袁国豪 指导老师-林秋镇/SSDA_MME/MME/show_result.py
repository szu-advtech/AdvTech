import matplotlib.pyplot as plt
import numpy as np

# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)


y_old_loss = [3.427811,0.022554,0.036347,0.035687,0.033419,0.034094,0.077897,0.037009,0.032475,0.031424,0.031363,0.031170,0.028953,0.036944,0.031145]  # loss值，即y轴
y_new_loss = [3.427811,0.023902,0.020967,0.026341,0.023782,0.018022,0.013932,0.021678,0.021163,0.021871,0.023263,0.016238,0.021939,0.018948,0.025421]  # loss值，即y轴
x_train_loss = [0,500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000]  # loss的数量，即x轴
x_tgt_loss=[500,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500,6000,6500,7000]
y_old_tgt=[1.7952,1.8939,1.9755,2.0202,2.0441,2.0584,2.0956,2.1139,2.1413,2.1590,2.2241,2.2390,2.2457,2.3250]
y_new_tgt=[1.7623,1.7190,1.7275,1.7248,1.7362,1.7459,1.7662,1.7659,1.7654,1.7955,1.7821,1.7921,1.7952,1.8067]

new_train_loss_path = "./new_train_loss.txt"  # 存储文件路径
new_train_Hloss_path = "./new_train_Hloss.txt"  # 存储文件路径
new_train_tgtloss_path = "./new_train_tgtloss.txt"  # 存储文件路径



new_y_train_loss = data_read(new_train_loss_path)  # loss值，即y轴
new_x_train_loss = range(len(new_y_train_loss))  # loss的数量，即x轴

new_y_train_Hloss = data_read(new_train_Hloss_path)  # loss值，即y轴
new_x_train_Hloss = range(len(new_y_train_Hloss))  # loss的数量，即x轴

new_y_train_tgtloss = data_read(new_train_tgtloss_path)  # loss值，即y轴
new_x_train_tgtloss = range(len(new_y_train_tgtloss))  # loss的数量，即x轴
plt.figure()

# 去除顶部和右边框框
ax = plt.axes()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.xlabel('iters')  # x轴标签
plt.ylabel('loss')  # y轴标签

# 以x_train_loss为横坐标，y_train_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
# 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
plt.plot(new_x_train_loss, new_y_train_loss, linewidth=1,color='black', linestyle="solid", label="new train loss")
plt.plot(new_x_train_Hloss, new_y_train_Hloss, linewidth=1,color='black', linestyle="solid", label="new train Hloss")
#plt.plot(new_x_train_tgtloss, new_y_train_tgtloss, linewidth=1,color='black', linestyle="solid", label="new train tgtloss")



plt.legend()
plt.title('Loss curve')
plt.show()
