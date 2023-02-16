#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy

import torch
import math


def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg


def getDis(globalW, w):
    sumDis = 0
    w_avg = copy.deepcopy(w)
    for i in w_avg.keys():
        sumDis += torch.norm(w[i] - globalW[i], 2)

    return pow(float(sumDis), 0.5)


def getP(s_k, s_k_i):
    if s_k_i == 0:
        return 0
    sum_s_k = 0
    for cur_s_k_i in s_k:
        sum_s_k += cur_s_k_i
    return s_k_i / sum_s_k


def getEk(N_D, s_k):
    sum = 0
    for i in range(N_D):
        p = getP(s_k, s_k[i])
        if p == 0:
            continue
        sum += p * math.log(p)
    return -1.0 * (1 / math.log(N_D)) * sum


# 根据熵权法取得当前指标的权重
def getWk(N_D, s, s_i):
    sum = 0
    for i in range(len(s)):
        sum += getEk(N_D, s[i])
    return (1 - getEk(N_D, s_i)) / (len(s) - sum)


def getTauI(i, N_D, s):
    sum = 0
    for k in range(len(s)):
        sum += getWk(N_D, s, s[k]) * s[k][i]
    return sum


# 根据牛顿冷却法取得当前模型的权重
def getR(t, t0, theta, R0):
    return R0 * pow(math.e, -1 * (theta * (t - t0)))


def normalization(s):
    res = []
    for k in range(len(s)):
        res.append([])
        # min = s[k][0]
        # max = s[k][0]
        # for i in range(len(s[k])):
        #     if s[k][i] < min:
        #         min = s[k][i]
        #     if s[k][i] > max:
        #         max = s[k][i]
        sum = 0
        for i in range(len(s[k])):
            sum += s[k][i]
        if sum == 0:
            return 0
        # for i in range(len(s[k])):
        #     if max == min:
        #         res[k].append(1/len(s[k]))
        #         continue
        #     res[k].append((s[k][i] - min) / (max - min))
        for i in range(len(s[k])):
            res[k].append(s[k][i] / sum)
    return res


def getAlpha(kexi, t, t0, theta, R0, i, N_D, s):
    # 归一化解决时间戳数值过大导致的熵权过小的问题。
    s = normalization(s)
    return kexi * getR(t, t0, theta, R0) * getTauI(i, N_D, s)
