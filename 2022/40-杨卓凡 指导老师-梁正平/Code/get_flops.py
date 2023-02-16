from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

from math import ceil

import torch
from torch.autograd import Variable
from functools import reduce
import operator


# Code for "AMC: AutoML for Model Compression and Acceleration on Mobile Devices"
# Yihui He*, Ji Lin*, Zhijian Liu, Hanrui Wang, Li-Jia Li, Song Han
# {jilin, songhan}@mit.edu

count_ops = 0
count_params = 0


# 计算子层的数量
def get_num_gen(gen):
    return sum(1 for x in gen)


# 通过是否有子层，判断是否为叶子结点
def is_leaf(model):
    return get_num_gen(model.children()) == 0


# 得到该层的名字
def get_layer_info(layer):
    layer_str = str(layer)
    # print(layer_str)
    type_name = layer_str[:layer_str.find('(')].strip()
    return type_name


# 得到每层的参数量
def get_layer_param(model, compress_rate=1, is_conv=True):
    # 卷积层
    if is_conv:
        total = 0.
        for idx, param in enumerate(model.parameters()):
            assert idx < 2
            f = param.size()[0]
            stay_num = ceil(compress_rate * f)
            # torch.numel()函数，可以计算出单个tensor元素的个数
            total += stay_num * param.numel() / f
        return total
    # 线性层
    else:
        # 依次对线性中的每一层进行乘法
        return sum([reduce(operator.mul, i.size(), 1) for i in model.parameters()])


# 计算每层的计算量，并调用get_layer_param得到参数量
# x为该层输入的参数
def measure_layer(layer, x, compress_rate=1, print_name=False):
    # 引用全局变量
    global count_ops, count_params

    # 当前层的ops和params
    delta_ops = 0
    delta_params = 0

    # 乘法算1次运算
    multi_add = 1

    # 得到当前层的名字
    type_name = get_layer_info(layer)

    # 如果是卷积层
    if type_name in ['Conv2d', 'Conv2dDynamicSamePadding', 'Conv2dStaticSamePadding', 'MaskConv2d']:
        # 根据公式计算输出的C、W、H
        out_h = int((x.size()[2] + 2 * layer.padding[0] - layer.kernel_size[0]) /
                    layer.stride[0] + 1)
        out_w = int((x.size()[3] + 2 * layer.padding[1] - layer.kernel_size[1]) /
                    layer.stride[1] + 1)
        stay_num = ceil(compress_rate * layer.out_channels)
        # 现在的计算量
        delta_ops = layer.in_channels * stay_num * layer.kernel_size[0] * \
                    layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add
        # 原本的计算量
        delta_ops_ori = layer.in_channels * layer.out_channels * layer.kernel_size[0] * \
                        layer.kernel_size[1] * out_h * out_w / layer.groups * multi_add

        # 得到参数量
        delta_params = get_layer_param(layer, compress_rate)

        # 输出每层的参数量
        print(compress_rate, [layer.out_channels, layer.in_channels, layer.kernel_size[0], layer.kernel_size[1]],
              'params:', delta_params, ' flops:', delta_ops_ori)

    # 线性层
    elif type_name in ['Linear']:

        # 计算计算量与参数量
        weight_ops = layer.weight.numel() * multi_add
        bias_ops = layer.bias.numel()
        delta_ops = x.size()[0] * (weight_ops + bias_ops)
        delta_params = get_layer_param(layer, compress_rate, is_conv=False)

        print('linear:', layer, delta_ops, delta_params)

    elif type_name in ['DenseBasicBlock', 'ResBasicBlock']:
        measure_layer(layer.conv1, x)

    elif type_name in ['Inception']:
        measure_layer(layer.conv1, x)

    elif type_name in ['DenseBottleneck', 'SparseDenseBottleneck']:
        measure_layer(layer.conv1, x)

    elif type_name in ['Transition', 'SparseTransition']:
        measure_layer(layer.conv1, x)

    elif type_name in ['ReLU', 'BatchNorm1d', 'BatchNorm2d', 'Dropout2d', 'DropChannel', 'Dropout', 'AdaptiveAvgPool2d',
                       'AvgPool2d', 'MaxPool2d', 'Mask', 'channel_selection', 'LambdaLayer', 'Sequential']:
        return
    # 未知的层
    else:
        raise TypeError('unknown layer type: %s' % type_name)
    
    # 全局变量加上本层的量
    count_ops += delta_ops
    count_params += delta_params
    return


# 对模型进行测量
def measure_model(model, device, C, H, W, compress_rate=1, print_name=False):
    # 引用全局变量并初始化
    global count_ops, count_params
    count_ops = 0
    count_params = 0

    # 自己创建个全零数据，计算
    data = Variable(torch.zeros(1, C, H, W)).to(device)
    model = model.to(device)

    # 切换到测试模式
    model.eval()

    # 判断是否计算，根据他是不是叶子节点
    def should_measure(x):
        return is_leaf(x)

    # 修改前向传播，并保存原本的
    def modify_forward(model, print_name):
        # 对每一个子层遍历
        for child in model.children():
            # 判断是否测量
            if should_measure(child):
                # 生成新的前向传播
                def new_forward(m):
                    # 遍历该层，改造
                    def lambda_forward(x):
                        measure_layer(m, x, compress_rate, print_name)
                        return m.old_forward(x)

                    return lambda_forward

                child.old_forward = child.forward
                child.forward = new_forward(child)
            # 递归遍历
            else:
                modify_forward(child, print_name)

    # 恢复原本的前向传播
    def restore_forward(model):
        # child只获取模型‘儿子’，不再往深处获取’孙子
        for child in model.children():
            # 如果是叶子结点并且已经改造
            if is_leaf(child) and hasattr(child, 'old_forward'):
                child.forward = child.old_forward
                child.old_forward = None
            # 递归寻找
            else:
                restore_forward(child)

    # 修改模型
    modify_forward(model, print_name)
    # 模型计算
    model.forward(data)
    # 恢复模型
    restore_forward(model)

    return count_ops, count_params
