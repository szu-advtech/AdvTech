#!/usr/bin/python3
# coding=utf-8

import os, sys

sys.path.append('/data2/zengleilei/code/medical_code/rework/MSNet-main')
import datetime
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda import amp
from utils import dataset_medical
from model.miccai_msnet import MSNet, LossNet


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


use_fp16 = True
scaler = amp.GradScaler(enabled=use_fp16)


torch.cuda.set_device(0)


def train(Dataset, Network, Network1):
    ## dataset
    train_path = '/data2/zengleilei/code/medical_code/rework/data/TrainDataset'
    cfg = Dataset.Config(datapath=train_path, savepath='./saved_model/msnet', mode='train', batch=16, lr=0.05,
                         momen=0.9, decay=5e-4, epoch=50)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, num_workers=8)
    if not os.path.exists(cfg.savepath):
        os.makedirs(cfg.savepath)
    ## network
    net = Network()
    net1 = Network1()
    net.train(True)
    net1.eval()
    # net.cuda()
    net.cuda()
    net1.cuda()
    # net = torch.nn.DataParallel(net, device_ids=[2, 3])
    # net1 = torch.nn.DataParallel(net1, device_ids=[2, 3])
    # net1.cuda()
    torch.backends.cudnn.enabled = False  # res2net does not support cudnn in py17
    for param in net1.parameters():
        param.requires_grad = False
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone.conv1' in name or 'bkbone.bn1' in name:
            print(name)
        elif 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    global_step = 0

    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.1
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr

        for step, (image, mask) in enumerate(loader):
            image, mask = image.cuda().float(), mask.cuda().float()
            # image, mask = image.float(), mask.float()
            with amp.autocast(enabled=use_fp16):
                output = net(image)
                loss2u = net1(F.sigmoid(output), mask)
                loss1u = structure_loss(output, mask)
                # 改变loss 的权重 因为前期的提取特征能力并不好，但是vgg 是预训练过后的因为加大loss权重
                # 后期加大网络的bce 以及dice loss 权重
                loss = loss1u + 0.1 * loss2u
                # if epoch <= 5:
                #     # loss = loss1u + 0.1 * loss2u
                #     loss = 0.8 * loss1u + loss2u
                #     # loss = 0.5 * loss1u + loss2u
                # else:
                #     loss = loss1u + 0.1 * loss2u
            optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            if step % 10 == 0:
                print('%s | step:%d/%d/%d | lr=%.6f | loss1u=%.6f | loss2u=%.6f ' % (
                    datetime.datetime.now(), global_step, epoch + 1, cfg.epoch, optimizer.param_groups[0]['lr'],
                    loss1u.item(), loss2u.item()))

        if epoch > cfg.epoch / 3 * 2:
            torch.save(net.state_dict(), cfg.savepath + '/new_origin2_model-' + str(epoch + 1))


if __name__ == '__main__':
    train(dataset_medical, MSNet, LossNet)
