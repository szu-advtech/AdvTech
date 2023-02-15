# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/12/6 20:17
 @desc:
"""
import torch


def train(model, optimizer, x, y, lr=0.01):
    for p in optimizer.param_groups:
        p['lr'] = lr
    model.train()
    optimizer.zero_grad()
    y_ = model(x)
    loss = torch.nn.functional.cross_entropy(y_, y)
    loss.backward()
    optimizer.step()
    return loss, y_


def test(model, x, y):
    x = x.cuda()
    y = y.cuda()
    model.eval()
    y_ = model(x)
    loss = torch.nn.functional.cross_entropy(y_, y)
    corrects = (torch.argmax(y_, dim=1).data == y.data)
    acc = corrects.cpu().int().sum().numpy()
    return loss, acc


def learning_rate_scheduler(epoch, lr, decay):
    if epoch >= 14:
        lr = decay*(((epoch-14)//5)+1)
    return lr
