# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/15 21:09
 @name: 
 @desc:
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from data_load.dataset import EEGImagesDataset
from model.conv_transformer import ConvTransformer
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from utils import train, test, learning_rate_scheduler

torch.manual_seed(1234)
np.random.seed(1234)

batch_size = 64
learning_rate = 0.0001
decay = 0.170
epochs = 40
k = 10
exp_id = '2022-12-6'

dataset = EEGImagesDataset(path='E:/Datasets/Stanford_digital_repository/img_pkl')
k_fold = KFold(n_splits=k, shuffle=True)
# loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=6, shuffle=True)

if __name__ == '__main__':
    global_step = 0
    for fold, (train_ids, valid_ids) in enumerate(k_fold.split(dataset)):
        train_sampler = SubsetRandomSampler(train_ids)
        valid_sampler = SubsetRandomSampler(valid_ids)
        train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=3,
                                  prefetch_factor=2)
        valid_loader = DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler, num_workers=1,
                                  prefetch_factor=1)
        n_t = len(train_ids)
        n_v = len(valid_loader)
        print('Fold -', fold, ' num of train and test: ', n_t, n_v)

        model = ConvTransformer(num_classes=6, channels=8, num_heads=2, E=16, F=256, T=32, depth=2).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, betas=(0.9, 0.98), eps=1e-9)
        summary = SummaryWriter(log_dir='./log/' + exp_id + '/' + str(fold) + '_fold/')

        for epoch in range(epochs):
            for step, (x, y) in enumerate(train_loader):
                x = x.cuda()
                y = y.cuda()
                lr = learning_rate_scheduler(epoch=epoch, lr=learning_rate, decay=0.5)
                loss, y_ = train(model=model, optimizer=optimizer, x=x, y=y, lr=lr)
                global_step += 1
                if step % 50 == 0:
                    corrects = (torch.argmax(y_, dim=1).data == y.data)
                    acc = corrects.cpu().int().sum().numpy() / batch_size
                    summary.add_scalar(tag='TrainLoss', scalar_value=loss, global_step=global_step)
                    summary.add_scalar(tag='TrainAcc', scalar_value=acc, global_step=global_step)
                    print('epoch:{}/{} step:{}/{} global_step:{} lr:{:.5f} loss={:.5f} acc={:.3f}'.format(
                        epoch, epochs, step, int(n_t / batch_size), global_step, lr, loss, acc))
        print('Training done')
        for step, (x, y) in enumerate(valid_loader):
            loss, acc = test(model=model, x=x, y=y)
            acc = acc / batch_size
            summary.add_scalar(tag='ValLoss', scalar_value=loss, global_step=global_step)
            summary.add_scalar(tag='ValAcc', scalar_value=acc, global_step=global_step)
            print('test step:{}/{} loss={:.5f} acc={:.3f}'.format(step, int(n_v / batch_size), loss, acc))
        print('Testing done')
