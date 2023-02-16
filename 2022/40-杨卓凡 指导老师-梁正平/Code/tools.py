import os
import re
import numpy as np
import torch

from num import feature_result_get, total_get, total_set, feature_result_set, feature_result_init, total_init


# 得到每层的特征图
def get_feature_hook(self, input, output):
    feature_result = feature_result_get()
    total = total_get()

    a = output.shape[0]
    b = output.shape[1]
    # 低版本
    c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b)])
    # # 高版本
    # c = torch.tensor([torch.linalg.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b)])

    c = c.view(a, -1).float()
    c = c.sum(0)
    feature_result = feature_result * total + c
    total = total + a
    total_set(total)
    feature_result = feature_result / total
    feature_result_set(feature_result)


# 使用limit次的batch得到rank值
def feature_test(net, limit, device, trainloader):
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(trainloader):
            if i >= limit:
                break
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)


# 得到裁剪需要的秩并保存
def get_feature(net, limit, device, trainloader):
    cfg = net.relucfg

    # 创建存储文件夹
    if not os.path.isdir('rank_conv'):
        os.mkdir('rank_conv')

    # 遍历每个卷积层
    for i, cov_id in enumerate(cfg):
        feature_result_init()
        total_init()
        cov_layer = net.features[cov_id]
        handler = cov_layer.register_forward_hook(get_feature_hook)
        feature_test(net, limit, device, trainloader)
        handler.remove()
        np.save('rank_conv/rank_conv' + str(i + 1) + '.npy', feature_result_get().numpy())


# 测试
def test(net, testloader, device):
    # 统计个数
    total, correct = 0, 0

    with torch.no_grad():
        for _, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    print('Acc:%.5f' % (1.0 * correct / total))


# 训练并依次裁剪每一层
def fine_train(net, prun_net, cov_id, optimizer, criterion, device, trainloader, pruning=True):
    net.train()
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        with torch.cuda.device(device):
            inputs = inputs.to(device)
            targets = targets.to(device)
            # 清空过往梯度
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            # 计算损失
            loss.backward()

            # 更新模型
            optimizer.step()

            # 去掉已经被删除的filter
            if pruning:
                prun_net.grad_mask(cov_id)


# 训练函数
def train(net, train_loader, EPOCHS, optimizer, criterion, device, scheduler):
    # 将由CPU保存的模型加载到GPU上
    net.to(device)

    net.train()

    # 开始训练
    for epoch in range(EPOCHS):
        for _, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            # 将梯度信息设置为0
            optimizer.zero_grad()
            # 前向传播
            outputs = net(inputs)
            # 计算损失
            loss = criterion(outputs, targets)
            # 反向传播
            loss.backward()
            # 更新参数
            optimizer.step()
        # 更新学习率
        scheduler.step()


# HRank
def HRank(net, prun_net, optimizer, scheduler, criterion, trainloader, device):
    # 依次裁剪每一个卷积层
    for cov_id in range(len(net.convcfg)):

        # 计算每一个剪枝卷积层需要裁剪的filter
        prun_net.layer_mask(cov_id + 1, 4)

        # 剪枝1次，重新训练2次
        for epoch in range(0, 2):
            fine_train(net, prun_net, cov_id, optimizer, criterion, device, trainloader)
            scheduler.step()
        train(net, trainloader, 5, optimizer, criterion, device, scheduler)
    return
