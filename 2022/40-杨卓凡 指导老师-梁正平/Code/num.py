import torch

# 全局变量
feature_result = torch.tensor(0.)
total = torch.tensor(0.)


def feature_result_init():
    global feature_result
    feature_result = torch.tensor(0.)


def total_init():
    global total
    total = torch.tensor(0.)


def feature_result_get():
    global feature_result
    return feature_result


def total_get():
    global total
    return total


def feature_result_set(res):
    global feature_result
    feature_result = res


def total_set(res):
    global total
    total = res
