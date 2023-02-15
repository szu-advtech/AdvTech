#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 13:33:52 2021

@author: wjz
"""
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

from model import BTICSwin
import trainer
import dataset

data_path = './fakeddit'
save_path = './model'

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main_func():
    # data
    # image = dataset.image
    text = dataset.text

    print("creating dataset")
    train_dataset = dataset.SimDataset("train")
    valid_dataset = dataset.SimDataset("valid")
    test_dataset = dataset.SimDataset("test")

    train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True, pin_memory=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=True, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16)
    print("finish loading dataset")

    # # Field
    # label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.float)
    # id_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.int)
    # # 看起来并不需要t11，只需要记录id即可
    # fields = [('Id', id_field), ('label', label_field),
    #           ('11i', id_field), ('12i', id_field), ('13i', id_field), ('14i', id_field), ('15i', id_field),
    #           ('01i', id_field), ('02i', id_field), ('03i', id_field), ('04i', id_field), ('05i', id_field)]
    #
    # # 用于训练的数据是仅含 id 的表，即 Id i01 ...
    # # text 和 image 是完整表，用 id 来查
    # train, valid, test = TabularDataset.splits(path=data_path, train='train_a_new.csv', validation='valid_a_new.csv',
    #                                            test='test_a_new.csv', format='CSV', fields=fields, skip_header=True)
    #
    # # Iterators
    # train_iter = BucketIterator(train, batch_size=8, sort_key=lambda x: int(x.Id), shuffle=True,
    #                             device=device, train=True, sort=True, sort_within_batch=True)
    #
    # valid_iter = BucketIterator(valid, batch_size=8, sort_key=lambda x: int(x.Id), shuffle=True,
    #                             device=device, train=True, sort=True, sort_within_batch=True)
    # test_iter = Iterator(test, batch_size=8, device=device, train=False, shuffle=False, sort=False)

    # model = BTICSwin(text, device).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=1e-6)
    # trainer.train(model, optimizer, device, train_loader=train_dataloader, valid_loader=valid_dataloader, file_path=save_path)
    # trainer.visualize_train_process(save_path, device)

    best_model = BTICSwin(text, device).to(device)
    trainer.load_checkpoint(f'{save_path}/model_BTIC_Swin.pth', best_model, device)
    trainer.evaluate(best_model, test_dataloader, device, save_path)


if __name__ == '__main__':
    main_func()
