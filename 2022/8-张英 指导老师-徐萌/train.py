import torch
import torch.nn as nn
import torch.nn.functional as F
import os
# from SG_model import HSI_Dehaze_Model
from SGNet2 import SGNet
from gunet import gunet_s
from load_data import train_data_loader
import matplotlib.pyplot as plt

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)  # cuda

# 数据的通道数
channel = 150

epoch_num = 201
# 学习率
lr = 0.001
batch_size = 32

net = SGNet(channel).to(device=device)

# 损失函数
loss_func = nn.MSELoss()

# 优化器
optimizer = torch.optim.Adam(net.parameters(), lr=lr)  # pytorch Adam没有momentum参数

# optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)   # 采用随机梯度下降


# 调整学习率
# step_size=5:每进行5个epoch调整一次学习率，  gamma=0.9,每次调整为上一次的0.9倍
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
#                                             step_size=5,
#                                             gamma=0.9)

train_losses = []
# test_losses = []
for epoch in range(epoch_num):
    # print("epoch is", epoch)  # 第几个epoch
    net.train()
    tamp = 0
    j = 0
    for i, data in enumerate(train_data_loader):
        j = j+1
        # print("step ", i)  # 一个epoch 第几个迭代次数(第几个batch)
        # print(len(data))
        inputs, labels = data  # 输入数据和标签
        inputs, labels = inputs.to(device=device), labels.to(device=device)
        # print("inputs.shape", inputs.shape)       # [b,c,h,w]
        # print("labels.shape", labels.shape)

        outputs = net(inputs)  # 将输入放进网络，得到预测输出结果
        loss = loss_func(outputs, labels)  # 预测输出和真实的labels,计算损失
        tamp = tamp + loss.item()
        # print("loss", loss.item())

        optimizer.zero_grad()  # 反向传播前，将优化器的梯度置为0，否则梯度会累加
        loss.backward()  # 反向传播
        optimizer.step()  # 更新参数
    train_losses.append(tamp/j)
    print("epoch", epoch + 1,"train_loss is", tamp/j)  # 打印一个epoch的平均loss

    if not os.path.exists("models"):
        os.mkdir("models")

    if epoch % 25 == 0:
        torch.save(net.state_dict(), r"./models/{}.pth".format(epoch))  # 每10epoch保存一个模型
        print('save successful')

###
x = [i for i in range(epoch_num)]
plt.plot(x,train_losses)
# plt.plot(x,test_losses)
plt.savefig('./1.png')
plt.show()




