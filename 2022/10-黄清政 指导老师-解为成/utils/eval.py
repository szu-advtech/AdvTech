from __future__ import print_function, absolute_import

__all__ = ['accuracy']

import torch
import torch.nn as nn


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""

    maxk = max(topk)

    #返回最大值和位置索引
    #返回k个最大元素
    data, pred = output.topk(maxk, 1,largest = True, sorted = True)
    #转置
    pred = pred.t()
    #expand_as扩展为pred的shape

    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = (correct == True).sum()

    return res
