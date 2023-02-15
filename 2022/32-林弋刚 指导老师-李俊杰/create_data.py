# -*- coding: utf-8 -*-
"""
这部分程序可以将放在一起的图片按照训练集测试集和验证集分开
但是有个问题
因为这个数据集是从imagenet抽取出来的
训练集测试集、测试集和验证集的数据类别都不相同
其类别总数是100类
这意味着我们不能从头开始训练
因此还需要对这个数据集进行重新划分
"""

import csv
import os
from PIL import Image

train_csv_path = "E:/BaiduNetdiskDownload/mini-imagenet/train.csv"
val_csv_path = "E:/BaiduNetdiskDownload/mini-imagenet/val.csv"
test_csv_path = "E:/BaiduNetdiskDownload/mini-imagenet/test.csv"

train_label = {}
val_label = {}
test_label = {}
with open(train_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        train_label[row[0]] = row[1]

with open(val_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        val_label[row[0]] = row[1]

with open(test_csv_path) as csvfile:
    csv_reader = csv.reader(csvfile)
    birth_header = next(csv_reader)
    for row in csv_reader:
        test_label[row[0]] = row[1]

img_path = "E:/BaiduNetdiskDownload/mini-imagenet/mini-imagenet/images"
new_img_path = "E:/BaiduNetdiskDownload/mini-imagenet/mini-imagenet/images_OK"
for png in os.listdir(img_path):
    path = img_path + '/' + png
    im = Image.open(path)
    if (png in train_label.keys()):
        tmp = train_label[png]
        temp_path = new_img_path + '/train' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)
        # with open(temp_path, 'wb') as f:
        #     f.write(path)

    elif (png in val_label.keys()):
        tmp = val_label[png]
        temp_path = new_img_path + '/val' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)

    elif (png in test_label.keys()):
        tmp = test_label[png]
        temp_path = new_img_path + '/test' + '/' + tmp
        if (os.path.exists(temp_path) == False):
            os.makedirs(temp_path)
        t = temp_path + '/' + png
        im.save(t)
