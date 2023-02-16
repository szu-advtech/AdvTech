import networkx as nx
import random
import math

import Util as util


def RW(G=nx.Graph(), P=0.2):
    if len(G.nodes) * P < 1:
        return nx.Graph()

    GS = nx.Graph()

    i = random.sample(list(G.nodes), 1)[0]  # 出发结点

    m = len(G.nodes) * 10  # 迭代步数
    step = 0
    while step <= m:
        j = random.sample(list(G.neighbors(i)), 1)[0]
        i = j
        step += 1

    # 开始随机游走抽样

    node_set = set([])
    node_set.add(i)
    GS.add_node(i)
    while len(node_set) <= len(G.nodes) * P:
        j = random.sample(list(G.neighbors(i)), 1)[0]
        node_set.add(j)
        GS.add_edge(i, j)
        i = j

    for node in node_set:
        GS.add_node(node)

    for u in GS.nodes:
        for v in GS.nodes:
            if G.has_edge(u, v):
                GS.add_edge(u, v)
    return GS


def RJ(G=nx.Graph(), P=0.2):
    GS = nx.Graph()

    i = random.sample(list(G.nodes), 1)[0]  # 出发结点
    d = 0.15  # 迁跃概率
    # 开始随机游走抽样
    node_set = set([])
    node_set.add(i)
    GS.add_node(i)
    while len(node_set) <= len(G.nodes) * P:
        if random.random() <= d:
            j = random.sample((list(G.nodes)), 1)[0]
        else:
            if G.degree[i] == 0:
                continue
            j = random.sample(list(G.neighbors(i)), 1)[0]
        node_set.add(j)
        if G.has_edge(i, j):
            GS.add_edge(i, j)
        else:
            GS.add_node(j)
        i = j

    for node in node_set:
        GS.add_node(node)

    for u in GS.nodes:
        for v in GS.nodes:
            if G.has_edge(u, v):
                GS.add_edge(u, v)
    return GS


def Meropolis_Hastings_RW(G=nx.Graph(), P=0.2):
    GS = nx.Graph()

    m = len(G.nodes) * 10  # 迭代步数
    step = 0

    i = random.sample(list(G.nodes), 1)[0]  # 出发结点

    # 迭代m次
    while step <= m:
        j = random.sample(list(G.neighbors(i)), 1)[0]
        a = 1 if 1 < G.degree[i] / G.degree[j] else G.degree[i] / G.degree[j]
        if random.random() < a:
            i = j
            step += 1
        else:
            continue

    step = 0
    step_max = len(G.nodes) / 10
    node_set = set([])
    node_set_list = []

    # 判断马尔可夫链是否达到收敛状态
    while 1:
        # 抽取到10组顶点，判断每组顶点个数是否收敛
        if len(node_set_list) == 10:
            mean1 = 0
            mean2 = 0
            for k in range(0, 5):
                mean1 += len(node_set_list[k])
            for k in range(5, 10):
                mean2 += len(node_set_list[k])

            # print(f"{mean1 / 5}-{mean2 / 5}-{step_max}")
            # print(f"{math.fabs(mean1 / 5 - mean2 / 5)}-----{step_max * 0.05}")

            if math.fabs(mean1 / 5 - mean2 / 5) < step_max * 0.05:  # 达到了收敛状态
                break
            else:
                node_set_list = []
                step = 0
                node_set = set([])

        if step > step_max:
            node_set_list.append(node_set)
            step = 0
            node_set = set([])

        j = random.sample(list(G.neighbors(i)), 1)[0]
        a = 1 if 1 < G.degree[i] / G.degree[j] else G.degree[i] / G.degree[j]

        if random.random() < a:
            i = j
            node_set.add(i)
            step += 1
        else:
            continue

    # 开始抽样
    node_set = set([])
    node_set.add(i)
    while len(node_set) <= len(G.nodes) * P:
        j = random.sample(list(G.neighbors(i)), 1)[0]
        a = 1 if 1 < G.degree[i] / G.degree[j] else G.degree[i] / G.degree[j]

        if random.random() < a:
            GS.add_edge(i, j)
            i = j
            node_set.add(i)
        else:
            continue

    for node in node_set:
        GS.add_node(node)

    for u in GS.nodes:
        for v in GS.nodes:
            if G.has_edge(u, v):
                GS.add_edge(u, v)

    return GS


def Meropolis_Hastings_RJ(G=nx.Graph(), P=0.2):
    GS = nx.Graph()

    m = len(G.nodes) * 10  # 迭代步数
    step = 0

    i = random.sample(list(G.nodes), 1)[0]  # 出发结点
    d = 0.15  # 迁跃概率

    # 迭代m次
    while step <= m:
        if random.random() <= d:
            j = random.sample((list(G.nodes)), 1)[0]  # 迁跃
            if G.has_edge(i, j):
                tempP = ((1 - d) / len(G.nodes) + d / G.degree[j]) / ((1 - d) / len(G.nodes) + d / G.degree[i])
                a = 1 if 1 < tempP else tempP
            else:
                a = 1 - d
        else:
            if G.degree[i] == 0:
                continue
            j = random.sample(list(G.neighbors(i)), 1)[0]
            tempP = ((1 - d) / len(G.nodes) + d / G.degree[j]) / ((1 - d) / len(G.nodes) + d / G.degree[i])
            a = 1 if 1 < tempP else tempP
        if random.random() < a:
            i = j
            step += 1
        else:
            continue

    step = 0
    step_max = len(G.nodes) / 10
    node_set = set([])
    node_set_list = []

    # 判断马尔可夫链是否达到收敛状态
    while 1:
        # 抽取到10组顶点，判断每组顶点个数是否收敛
        if len(node_set_list) == 10:
            mean1 = 0
            mean2 = 0
            for k in range(0, 5):
                mean1 += len(node_set_list[k])
            for k in range(5, 10):
                mean2 += len(node_set_list[k])

            # print(f"{mean1 / 5}-{mean2 / 5}-{step_max}")
            # print(f"{math.fabs(mean1 / 5 - mean2 / 5)}-----{step_max * 0.05}")

            if math.fabs(mean1 / 5 - mean2 / 5) < step_max * 0.05:  # 达到了收敛状态
                break
            else:
                node_set_list = []
                step = 0
                node_set = set([])

        if step > step_max:
            node_set_list.append(node_set)
            step = 0
            node_set = set([])

        if random.random() <= d:
            j = random.sample((list(G.nodes)), 1)[0]  # 迁跃
            if G.has_edge(i, j):
                tempP = ((1 - d) / len(G.nodes) + d / G.degree[j]) / ((1 - d) / len(G.nodes) + d / G.degree[i])
                a = 1 if 1 < tempP else tempP
            else:
                a = 1 - d
        else:
            if G.degree[i] == 0:
                continue
            j = random.sample(list(G.neighbors(i)), 1)[0]
            tempP = ((1 - d) / len(G.nodes) + d / G.degree[j]) / ((1 - d) / len(G.nodes) + d / G.degree[i])
            a = 1 if 1 < tempP else tempP

        if random.random() < a:
            i = j
            node_set.add(i)
            step += 1
        else:
            continue

    # 开始抽样
    node_set = set([])
    node_set.add(i)
    while len(node_set) <= len(G.nodes) * P:
        if random.random() <= d:
            j = random.sample((list(G.nodes)), 1)[0]  # 迁跃
            if G.has_edge(i, j):
                tempP = ((1 - d) / len(G.nodes) + d / G.degree[j]) / ((1 - d) / len(G.nodes) + d / G.degree[i])
                a = 1 if 1 < tempP else tempP
            else:
                a = 1 - d
        else:
            if G.degree[i] == 0:
                continue
            j = random.sample(list(G.neighbors(i)), 1)[0]
            tempP = ((1 - d) / len(G.nodes) + d / G.degree[j]) / ((1 - d) / len(G.nodes) + d / G.degree[i])
            a = 1 if 1 < tempP else tempP

        if random.random() < a:
            if G.has_edge(i, j):
                GS.add_edge(i, j)
            i = j
            node_set.add(i)
        else:
            continue

    for node in node_set:
        GS.add_node(node)

    for u in GS.nodes:
        for v in GS.nodes:
            if G.has_edge(u, v):
                GS.add_edge(u, v)

    return GS


def RW_test(G=nx.Graph(), P=0.2):
    GS = nx.Graph()

    i = random.sample(list(G.nodes), 1)[0]  # 出发结点

    m = len(G.nodes) * 3  # 迭代步数
    step = 0
    while step <= m:
        j = random.sample(list(G.neighbors(i)), 1)[0]
        i = j
        step += 1

    step = 0
    # 开始随机游走抽样
    node_set = set([])
    node_set.add(i)
    GS.add_node(i)
    while step <= len(G.nodes) * P:
        j = random.sample(list(G.neighbors(i)), 1)[0]
        node_set.add(j)
        GS.add_edge(i, j)
        i = j
        step += 1

    for node in node_set:
        GS.add_node(node)

    for u in GS.nodes:
        for v in GS.nodes:
            if G.has_edge(u, v):
                GS.add_edge(u, v)
    return GS


def Meropolis_Hastings_RW_test(G=nx.Graph(), P=0.2):
    GS = nx.Graph()

    m = len(G.nodes) * 10  # 迭代步数
    step = 0

    i = random.sample(list(G.nodes), 1)[0]  # 出发结点

    # 迭代m次
    while step <= m:
        j = random.sample(list(G.neighbors(i)), 1)[0]
        a = 1 if 1 < G.degree[i] / G.degree[j] else G.degree[i] / G.degree[j]
        if random.random() < a:
            i = j
            step += 1
        else:
            continue

    step = 0
    step_max = len(G.nodes) / 10
    node_set = set([])
    node_set_list = []

    # 判断马尔可夫链是否达到收敛状态
    while 1:
        # 抽取到10组顶点，判断每组顶点个数是否收敛
        if len(node_set_list) == 10:
            mean1 = 0
            mean2 = 0
            for k in range(0, 5):
                mean1 += len(node_set_list[k])
            for k in range(5, 10):
                mean2 += len(node_set_list[k])

            # print(f"{mean1 / 5}-{mean2 / 5}-{step_max}")
            # print(f"{math.fabs(mean1 / 5 - mean2 / 5)}-----{step_max * 0.05}")

            if math.fabs(mean1 / 5 - mean2 / 5) < step_max * 0.05:  # 达到了收敛状态
                break
            else:
                node_set_list = []
                step = 0
                node_set = set([])

        if step > step_max:
            node_set_list.append(node_set)
            step = 0
            node_set = set([])

        j = random.sample(list(G.neighbors(i)), 1)[0]
        a = 1 if 1 < G.degree[i] / G.degree[j] else G.degree[i] / G.degree[j]

        if random.random() < a:
            i = j
            node_set.add(i)
            step += 1
        else:
            continue

    step = 1
    # 开始抽样
    node_set = set([])
    node_set.add(i)
    while len(node_set) <= len(G.nodes) * P:
        j = random.sample(list(G.neighbors(i)), 1)[0]
        a = 1 if 1 < G.degree[i] / G.degree[j] else G.degree[i] / G.degree[j]

        if random.random() < a:
            GS.add_edge(i, j)
            i = j
            node_set.add(i)
            step += 1
        else:
            continue

    for node in node_set:
        GS.add_node(node)

    for u in GS.nodes:
        for v in GS.nodes:
            if G.has_edge(u, v):
                GS.add_edge(u, v)

    return GS
