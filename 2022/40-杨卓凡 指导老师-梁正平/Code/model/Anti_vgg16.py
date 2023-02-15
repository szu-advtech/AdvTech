import random
import torch
import numpy as np
import pickle

# 默认值
defaultcfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 512]
relucfg = [2, 6, 9, 13, 16, 19, 23, 26, 29, 33, 36, 39]
convcfg = [0, 3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]


class Anti_VGG:
    def __init__(self, model=None, compress_rate=None, device=None):
        self.param_per_cov = None
        self.model = model
        self.convcfg = convcfg
        self.relucfg = relucfg
        self.compress_rate = compress_rate
        self.mask = {}
        self.device = device

    def layer_mask(self, cov_id, param_per_cov=4):
        # Weights和Bias参数的值mask_vgg16.py
        params = self.model.parameters()
        prefix = "rank_conv/rank_conv"
        subfix = ".npy"

        self.param_per_cov = param_per_cov

        for index, item in enumerate(params):
            # 下一层的weight
            if index == cov_id * param_per_cov:
                break
            # 首先对卷积层的weight
            if index == (cov_id - 1) * param_per_cov:
                f, c, w, h = item.size()
                # 导入rank排序npy文件
                rank = np.load(prefix + str(cov_id) + subfix)
                # 确定要删除的filter数量
                pruned_num = int(self.compress_rate * f)
                # 排序，并保留大的
                # ind = np.argsort(rank)[pruned_num:]
                # 排序，并保留小的
                ind = np.argsort(-rank)[pruned_num:]
                # 随机保留
                # temp = np.arange(f)
                # np.random.shuffle(temp)
                # ind = temp[pruned_num:]

                # 将保留的过滤器赋1
                zeros = torch.zeros(f, 1, 1, 1).to(self.device)
                for i in range(len(ind)):
                    zeros[ind[i], 0, 0, 0] = 1.
                # 每层的mask
                self.mask[index] = zeros
                # data也进行同样的操作
                item.data = item.data * self.mask[index]

            if (cov_id - 1) * param_per_cov < index <= (cov_id - 1) * param_per_cov + param_per_cov - 1:
                # zero适用于后面的conv.bias、norm.weight、norm.bias
                self.mask[index] = torch.squeeze(zeros)
                item.data = item.data * self.mask[index]

    def grad_mask(self, cov_id):
        params = self.model.parameters()
        for index, item in enumerate(params):
            if index == cov_id * self.param_per_cov:
                break
            # 进行裁剪
            item.data = item.data * self.mask[index]
