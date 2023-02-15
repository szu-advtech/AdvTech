# import matplotlib.pyplot as plt
# import pandas as pd
# import torch
# from sklearn import manifold
# import numpy as np
#
#
# def visual(feat):
#     # t-SNE的最终结果的降维与可视化
#     ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
#
#     x_ts = ts.fit_transform(feat)
#
#     print(x_ts.shape)  # [num, 2]
#
#     x_min, x_max = x_ts.min(0), x_ts.max(0)
#
#     x_final = (x_ts - x_min) / (x_max - x_min)
#
#     return x_final
#
#
# # 设置散点形状
# maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# # 设置散点颜色
# colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
#           'hotpink']
# # 图例名称
# Label_Com = ['a', 'b', 'c', 'd']
# # 设置字体格式
# font1 = {'family': 'Times New Roman',
#          'weight': 'bold',
#          'size': 32,
#          }
#
#
# def plotlabels(S_lowDWeights, Trure_labels, name):
#     True_labels = Trure_labels.reshape((-1, 1))
#     S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
#     S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
#     print(S_data)
#     print(S_data.shape)  # [num, 3]
#
#     for index in range(3):  # 假设总共有三个类别，类别的表示为0,1,2
#         X = S_data.loc[S_data['label'] == index]['x']
#         Y = S_data.loc[S_data['label'] == index]['y']
#         plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)
#
#         plt.xticks([])  # 去掉横坐标值
#         plt.yticks([])  # 去掉纵坐标值
#
#     plt.title(name, fontsize=32, fontweight='normal', pad=20)
#
#
# feat = torch.rand(128, 1024)  # 128个特征，每个特征的维度为1024
# label_test1 = [0 for index in range(40)]
# label_test2 = [1 for index in range(40)]
# label_test3 = [2 for index in range(48)]
#
# label_test = np.array(label_test1 + label_test2 + label_test3)
# print(label_test)
# print(label_test.shape)
#
# fig = plt.figure(figsize=(10, 10))
#
# plotlabels(visual(feat), label_test, '(a)')
#
# plt.show()

import argparse

import torch
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet
from samplers import CategoriesSampler
from convnet import Convnet
from resnet12new import resnet12_wide
from utils import pprint, set_gpu, count_acc, Averager, euclidean_metric
import random
import numpy as np

seed_s=3401
random.seed(seed_s)
np.random.seed(seed_s)
torch.manual_seed(seed_s)
torch.cuda.manual_seed(seed_s)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--load', default='./save/proto-Conv4net-5way-1shot/max-acc.pth')
    parser.add_argument('--batch', type=int, default=1)#batch=1 test for t-SNE
    parser.add_argument('--way', type=int, default=5)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=30)
    args = parser.parse_args()
    pprint(vars(args))

    set_gpu(args.gpu)

    dataset = MiniImageNet('test')
    sampler = CategoriesSampler(dataset.label,args.batch, args.way, args.shot + args.query)
    loader = DataLoader(dataset, batch_sampler=sampler,num_workers=8, pin_memory=True)

    model = Convnet().cuda()
    #model = resnet12_wide().to('cuda')#新baseline下的resnet12
    model.load_state_dict(torch.load(args.load))
    model.eval()

    ave_acc = Averager()

    for i, batch in enumerate(loader, 1):
        data, _ = [_.cuda() for _ in batch]
        k = args.way * args.shot
        data_shot, data_query = data[:k], data[k:]

        x = model(data_shot)
        x = x.reshape(args.shot, args.way, -1).mean(dim=0)
        p = x

        get_query_vector=model(data_query)
        logits = euclidean_metric(model(data_query), p)
        label = torch.arange(args.way).repeat(args.query)
        label = label.type(torch.cuda.LongTensor)

        acc = count_acc(logits, label)
        ave_acc.add(acc)
        print('batch {}: {:.2f}({:.2f})'.format(i, ave_acc.item() * 100, acc * 100))
        return p, get_query_vector, label
        x = None; p = None; logits = None



proto,vector,label=main()

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np

def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)

    x_ts = ts.fit_transform(feat)

    #print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final

# 设置散点形状
maker = ['o', 's', '^', '*', 'p', 'D', 'h' ,'H', '<', '>', 'd' ]
# 设置散点颜色
colors = ['#e38c7a', '#656667', '#99a4bc', 'violet', 'olivedrab', 'hotpink']
# 图例名称
Label_Com = ['type1', 'type2', 'type3', 'type4', 'type5']

def plotlabels(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]

    for index in range(5):  #test5，五个类别
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.75)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值
        plt.xlabel('t-SNE dimension 1',fontsize=14)
        plt.ylabel('t-SNE dimension 2',fontsize=14)
        plt.legend(labels=Label_Com)

    plt.title(name, fontsize=18, fontweight='normal', pad=20)

def plotlabelsproto(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]

    for index in range(5):  #test5，五个类别
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=300, marker=maker[6], c=colors[5], edgecolors=colors[5], alpha=0.9)


all_points=torch.cat((proto,vector),0)
feat=all_points.cpu()
feat=feat.detach().numpy()
feat=visual(feat)

proto=feat[0:5]
other=feat[5:]
label_test=label.cpu()
labelproto=[0,1,2,3,4]

fig = plt.figure(figsize=(9,9))
plotlabels(other, label_test, '5way-1shot miniImagenet visualization')
plotlabelsproto(proto, np.array(labelproto), '5way-1shot miniImagenet visualization')

plt.show()
