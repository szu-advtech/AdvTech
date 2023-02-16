import graph
import numpy as np
from collections import defaultdict


# 采样节点的数据结构
class SampledData():
    def __init__(self, v):
        # 采样的节点
        self.index = v
        # 节点v的邻居节点集
        self.nlist = []


# 查询
def query(G: graph.Graph, v):
    # 初始化采样节点的数据结构
    data = SampledData(v)
    # 保存邻边
    data.nlist = list(G.nlist[v])
    return data


# 选择一个随机游走的起始点
def select_seed(G: graph.Graph):
    return np.random.choice(G.nodes)


# 随机游走进行采样
def random_walk(G, sample_size, seed):
    # 采样起始点
    v = seed
    # 保存采样的结果
    sampling_list = []
    # 用于查看节点是否被采样，使得能够采样到一定数量的不同节点集
    queried = defaultdict(int)
    # 未达到所需节点
    while len(queried) < sample_size:
        # 节点v的邻居节点
        data = query(G, v)
        # 将访问的节点v保存到queried字典中
        queried[v] = 1
        # 相应的邻居节点（采样数据）放入sampling_list中
        sampling_list.append(data)
        # 从节点v的邻居节点中随机选择一个节点作为下一次游走的节点v
        v = np.random.choice(data.nlist)

    return sampling_list
