import torchvision
from torchsummary import summary

import dnn_model
import torch
from torch import nn
from torchvision import datasets, transforms
import csv
import utils
import numpy as np

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_set = datasets.FashionMNIST('dataset/', download=False, train=True, transform=transform)
train_index = utils.non_iid_divide(train_set, 1, 8, 6000)
train_index = np.array(list((train_index[0]).astype(int)))
train_dataset = torch.utils.data.Subset(train_set, train_index)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=200, shuffle=True)

# 下载Fashion-MNIST测试集数据，并构建测试集数据载入器train_loader,每次从测试集中载入200张图片，每次载入都打乱顺序
test_set = datasets.FashionMNIST('dataset/', download=False, train=False, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=200, shuffle=True)

# 对训练集的全部数据学习15遍，这个数字越大，训练时间越长
epochs = 100
learning_rate = 0.01
model = dnn_model.CNNFashion_Mnist(nn.Module)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.5)


print(summary(model, input_size=(1, 28, 28)))
with open('log_result.txt', 'a') as f:
    f.write("\n学习率为" + str(learning_rate) + ", 训练" + str(epochs) + "轮\n")

# 将每次训练的训练误差和测试误差存储在这两个列表里，后面绘制误差变化折线图用
train_losses, test_losses, test_accuracy = [], [], []

print('开始训练')
for e in range(epochs):
    running_loss = 0

    # 本地训练一遍
    for images, labels in train_loader:
        # 将优化器中的求导结果都设为0，否则会在每次反向传播之后叠加之前的
        optimizer.zero_grad()
        # 对200张图片进行推断，计算损失函数，反向传播优化权重，将损失求和
        log_ps = model(images)
        loss = criterion(log_ps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # 每次学完一遍数据集，都进行以下测试操作
    else:
        test_loss = 0
        accuracy = 0
        # 测试的时候不需要开自动求导和反向传播
        with torch.no_grad():
            # 关闭Dropout
            model.eval()

            # 对测试集中的所有图片都过一遍
            for images, labels in test_loader:
                # 对传入的测试集图片进行正向推断、计算损失，accuracy为测试集一万张图片中模型预测正确率
                log_ps = model(images)
                test_loss += criterion(log_ps, labels)
                ps = torch.exp(log_ps)
                top_p, top_class = ps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)

                # 等号右边为每一批200张测试图片中预测正确的占比
                accuracy += torch.mean(equals.type(torch.FloatTensor))
        # 恢复Dropout
        model.train()
        # 将训练误差和测试误差存在两个列表里，后面绘制误差变化折线图用
        train_loss = running_loss / len(train_loader)
        test_loss = (test_loss / len(test_loader)).item()
        accuracy = (accuracy / len(test_loader)).item()
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        test_accuracy.append(accuracy)

        print("训练集学习次数: {}/{}.. ".format(e + 1, epochs),
              "训练误差: {:.3f}.. ".format(train_loss),
              "测试误差: {:.3f}.. ".format(test_loss),
              "模型分类准确率: {:.3f}".format(accuracy))

        with open('log_result.txt', 'a') as f:
            f.write("\n训练集学习次数: " + str(e + 1) + "/" + str(epochs) + "\n训练误差: " + str(
                train_loss) + "\n测试误差:" + str(test_loss) + "\n模型分类准确率:" + str(accuracy))

csv_file = open('result.csv', 'w', newline='', encoding='utf-8')
writer = csv.writer(csv_file)
writer.writerow(train_losses)
writer.writerow(test_losses)
writer.writerow(test_accuracy)
csv_file.close()

