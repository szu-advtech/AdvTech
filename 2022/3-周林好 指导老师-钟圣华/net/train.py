'''

训练VGG16模型

'''

import os
import time
import torch
import torchvision
from torchvision import transforms, utils
from torch.utils.data import DataLoader
from torch.autograd import Variable
from utils import utils
import matplotlib.pyplot as plt

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"



def restruct_model(model, num_cls, device):

    num_fc = model.classifier[6].in_features
    # 修改最后一层的输出维度，即分类数
    model.classifier[6] = torch.nn.Linear(num_fc, num_cls)

    # 对于模型的每个权重，使其不进行反向传播，即固定参数
    for param in model.parameters():
        param.requires_grad = False

    # 将分类器的最后输出层换成了num_cls，这一层需要重新学习
    for param in model.classifier[6].parameters():
        param.requires_grad = True
    # print(model)  # 查看模型结构
    model.to(device)
    return model


def dataload(trainData, valData, testData):
    """
    trainData: 训练集路径
    valData:验证集路径
    testData: 测试集路径
    """
    # 训练数据
    train_data = torchvision.datasets.ImageFolder(trainData, transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))
    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

    # 验证数据
    val_data = torchvision.datasets.ImageFolder(valData, transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))
    val_loader = DataLoader(val_data, batch_size=128, shuffle=True)

    # 测试数据
    test_data = torchvision.datasets.ImageFolder(testData, transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]))
    test_loader = DataLoader(test_data, batch_size=128, shuffle=True)

    return train_data, val_data, test_data, train_loader, val_loader, test_loader


def train(model, trainData, valData, testData):
    criterion = torch.nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)  # 优化器
    train_data, val_data, test_data, train_loader, val_loader, test_loader = dataload(trainData, valData, testData)

    log = []
    # 启动训练、验证
    epoches = 300
    for epoch in range(epoches):
        train_loss = 0.
        train_acc = 0.
        for step, data in enumerate(train_loader):
            batch_x, batch_y = data
            batch_x, batch_y = Variable(batch_x), Variable(batch_y)
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()  # GPU

            out = model(batch_x)
            loss = criterion(out, batch_y)
            train_loss += loss.item()

            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if step % 100 == 0:
            #     print('Epoch: ', epoch, 'Step', step,
            #           'Train_loss: ', train_loss / ((step + 1) * 20), 'Train acc: ', train_acc / ((step + 1) * 20))


        print('Epoch: ', epoch, 'Train_loss: ', train_loss / len(train_data), 'Train acc: ',
              train_acc / len(train_data))

        # 保存训练过程数据
        # info = dict()
        # info['Epoch'] = epoch
        # info['Train_loss'] = train_loss / len(train_data)
        # info['Train_acc'] = train_acc / len(train_data)
        # log.append(info)

        # 验证集
        val_loss = 0
        val_acc = 0
        model.eval()
        with torch.no_grad():
            for step, data in enumerate(val_loader):
                batch_x, batch_y = data
                batch_x, batch_y = Variable(batch_x), Variable(batch_y)
                batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
                out = model(batch_x)
                loss = criterion(out, batch_y)
                val_loss += loss.item()

                pred = torch.max(out, 1)[1]
                val_correct = (pred == batch_y).sum()
                val_acc += val_correct.item()

            print('Val_loss: ', val_loss / len(val_data), 'Val acc: ', val_acc / len(val_data))


        # 保存训练、验证过程数据
        info = dict()
        info['Epoch'] = epoch
        info['Train_acc'] = train_acc / len(train_data)
        info['Val_acc'] = val_acc / len(val_data)
        log.append(info)


    draw(log)

    # 模型保存
    model_without_ddp = model
    os.chdir('../model_data')
    dir_name = time.strftime('%m-%d-%Hh%Mm')
    os.mkdir(dir_name)
    utils.save_on_master({
        'model': model_without_ddp.state_dict()},
        os.path.join(dir_name, 'model.pth'))
    model.eval()

    os.chdir('../')
    eval_loss = 0
    eval_acc = 0
    for step, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x, batch_y = Variable(batch_x), Variable(batch_y)
        batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
        out = model(batch_x)
        loss = criterion(out, batch_y)
        eval_loss += loss.item()

        pred = torch.max(out, 1)[1]
        test_correct = (pred == batch_y).sum()
        eval_acc += test_correct.item()
    print('Test_loss: ', eval_loss / len(test_data), 'Test acc: ', eval_acc / len(test_data))


def draw(logs: list):
    plt.figure()
    epoch = []
    train_acc = []
    val_acc = []
    for log_ in logs:
        epoch.append(log_['Epoch'])
        train_acc.append(log_['Train_acc'])
        val_acc.append(log_['Val_acc'])
    plt.plot(epoch, train_acc, 'r-', label='train_acc')
    plt.plot(epoch, val_acc, 'b-', label='val_acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig('acc.png', bbox_inches='tight')


if __name__ == '__main__':
    device = 'cuda'
    model_vgg16 = torchvision.models.vgg16(pretrained=True)
    num_cls = 7  # 七个类别
    model = restruct_model(model_vgg16, num_cls, device)
    #
    train_data = 'C:/Users/Administrator/Desktop/Evalution/datasets/train'
    val_data = 'C:/Users/Administrator/Desktop/Evalution/datasets/val'
    test_data = 'C:/Users/Administrator/Desktop/Evalution/datasets/test'

    begin_time = time.time()
    train(model, train_data, val_data, test_data)
    t = time.time() - begin_time
    print('运行时间：{}h'.format(t / 3600))     # 迭代运行时间