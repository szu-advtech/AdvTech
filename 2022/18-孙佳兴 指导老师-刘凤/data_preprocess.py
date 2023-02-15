# CIA
# 孙佳兴

import numpy as np
import torch
from scipy.io import loadmat, savemat
import os
import spectral as spy

import math
import torch.nn as nn


# 高斯滤波器构造函数
def get_gaussian_kernel(kernel_size=9, sigma=2, channels=102):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(
        -torch.sum((xy_grid - mean) ** 2., dim=-1) / (2 * variance))

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, groups=channels,
                                bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


if __name__ == '__main__':
    rgb = {"pavia": [55, 30, 5]}
    set_name = "pavia"
    prefix = "../data/" + set_name
    results_prefix = "../results/" + set_name
    if not os.path.exists(results_prefix):
        os.makedirs(results_prefix)
    # 读取并转tensor
    print("读取并转tensor……")
    data = loadmat(prefix + "/Pavia.mat")['pavia']  # 读取Pavia数据集
    spy.save_rgb(prefix + "/Pavia.jpg", data, rgb[set_name])

    sub_data = data[:960, :640, :]  # 截取Pavia数据集子图像960*640
    # spy.save_rgb(results_prefix + "/complete.jpg", sub_data, rgb[set_name])

    # 归一化
    max, min = np.max(sub_data, axis=(0, 1)), np.min(sub_data, axis=(0, 1))
    data_normalization = (sub_data - min) / (max - min)
    spy.save_rgb(results_prefix + "/gtHS.jpg", data_normalization, rgb[set_name])
    data_normalization = data_normalization.transpose(2, 0, 1)

    savemat(prefix + "/gtHS.mat", {'b': np.array(data_normalization)})  # CHW

    gtHS_data = torch.DoubleTensor(data_normalization)  # 转Tensor
    # 高斯模糊
    print("高斯模糊……")
    gtHS_data_expand = gtHS_data.unsqueeze(0)  # 需要四维才能输入
    gaussian_filter = get_gaussian_kernel(kernel_size=9, sigma=2, channels=102).double()  # Pavia数据集构造高斯滤波器
    gaussian_blur_data = gaussian_filter(gtHS_data_expand)  # 高斯模糊后的数据

    # 下采样
    print("下采样……")
    maxpool2 = nn.MaxPool2d(4)  # 构建采样器
    LRHS_data = maxpool2(gaussian_blur_data)  # 下采样， 模拟退化低分辨率高光谱图像
    LRHS_data = LRHS_data.squeeze(0)
    savemat(prefix + "/LRHS.mat", {'b': np.array(LRHS_data)})  # CHW
    spy.save_rgb(results_prefix + "/LRHS.jpg", np.array(LRHS_data).transpose((1, 2, 0)), rgb[set_name])


    # PAN
    print("PAN生成……")
    PAN_data = torch.mean(gtHS_data[:79, :, :], 0, keepdim=True)  # Pavia数据集可见光波段平均
    savemat(prefix + "/PAN.mat", {'b': np.array(PAN_data)})

    '''
    数据切分，LRHS 102*40*40，PAN 160*160
    不重叠的平均切K1个，再在每个平均切好的块里按固定步长切重叠的符合上述大小的的K2个，test_data_num：测试数据编号（K1个大块中）
    '''
    print("数据切分……")
    # Pavia数据集
    K1, K2 = 12, 21
    K1_R = 3
    K1_C = 4
    test_data_num = [4, 6, 8]
    stride = 2
    corrent = 0
    train_num = 0
    test_num = 0
    for i in range(K1_R):
        for j in range(K1_C):
            print("第{}块非重叠子块正在切分…………………………已完成{}%".format(i * K1_C + j + 1,
                                                                       round(((i * K1_C + j) / (K1_R * K1_C)) * 100,
                                                                             2)))
            corrent = corrent + 1
            tmp_LRHS_data = LRHS_data[:, 80 * i:80 * (i + 1), 40 * j:40 * (j + 1)]
            tmp_PAN_data = PAN_data[:, 320 * i:320 * (i + 1), 160 * j:160 * (j + 1)]
            tmp_gtHS_data = gtHS_data[:, 320 * i:320 * (i + 1), 160 * j:160 * (j + 1)]
            tmp_LRHS_data_list = []
            tmp_PAN_data_list = []
            tmp_gtHS_data_list = []
            for step in range(K2):
                tmp_LRHS_data_list.append(tmp_LRHS_data[:, stride * step:stride * step + 40, :])
                tmp_PAN_data_list.append(tmp_PAN_data[:, stride * step:stride * step + 160, :])
                tmp_gtHS_data_list.append(tmp_gtHS_data[:, stride * step:stride * step + 160, :])

            if corrent in test_data_num:  # 测试数据集
                for num in range(K2):
                    if not os.path.exists(prefix + "/LRHS/test"):
                        os.makedirs(prefix + "/LRHS/test")
                    savemat(prefix + "/LRHS/test/LRHS_{}.mat".format(test_num * K2 + num + 1),
                            {'b': np.array(tmp_LRHS_data_list[num])})

                    if not os.path.exists(results_prefix + "/LRHS"):
                        os.makedirs(results_prefix + "/LRHS")
                    spy.save_rgb(results_prefix + "/LRHS/LRHS_{}.jpg".format(test_num * K2 + num + 1),
                                 np.array(tmp_LRHS_data_list[num]).transpose((1, 2, 0)), rgb[set_name])

                    if not os.path.exists(prefix + "/PAN/test"):
                        os.makedirs(prefix + "/PAN/test")
                    savemat(prefix + "/PAN/test/PAN_{}.mat".format(test_num * K2 + num + 1),
                            {'b': np.array(tmp_PAN_data_list[num])})

                    if not os.path.exists(prefix + "/gtHS/test"):
                        os.makedirs(prefix + "/gtHS/test")
                    savemat(prefix + "/gtHS/test/gtHS_{}.mat".format(test_num * K2 + num + 1),
                            {'b': np.array(tmp_gtHS_data_list[num])})

                    if not os.path.exists(results_prefix + "/gtHS"):
                        os.makedirs(results_prefix + "/gtHS")
                    spy.save_rgb(results_prefix + "/gtHS/gtHS_{}.jpg".format(test_num * K2 + num + 1),
                                 np.array(tmp_gtHS_data_list[num]).transpose((1, 2, 0)), rgb[set_name])

                test_num += 1
            else:  # 训练数据集
                for num in range(K2):
                    if not os.path.exists(prefix + "/LRHS/train"):
                        os.makedirs(prefix + "/LRHS/train")
                    savemat(prefix + "/LRHS/train/LRHS_{}.mat".format(train_num * K2 + num + 1),
                            {'b': np.array(tmp_LRHS_data_list[num])})
                    if not os.path.exists(prefix + "/PAN/train"):
                        os.makedirs(prefix + "/PAN/train")
                    savemat(prefix + "/PAN/train/PAN_{}.mat".format(train_num * K2 + num + 1),
                            {'b': np.array(tmp_PAN_data_list[num])})

                    if not os.path.exists(prefix + "/gtHS/train"):
                        os.makedirs(prefix + "/gtHS/train")
                    savemat(prefix + "/gtHS/train/gtHS_{}.mat".format(train_num * K2 + num + 1),
                            {'b': np.array(tmp_gtHS_data_list[num])})

                train_num += 1
    print("数据切分已完成。")


