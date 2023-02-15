from scipy import stats
import networkx as nx
import math


def scipyKS(sample1=[], sample2=[]):
    # sample1 = [1, 2, 3, 4, 5, 6]
    # sample2 = [2, 3, 3, 4, 5, 8]
    return stats.kstest(sample1, sample2)


def KolmogorowSmirnov(sample1=[], sample2=[]):
    D = 0
    # sample1 = [1.2, 1.4, 1.9, 3.7, 4.4, 4.8, 9.7, 17.3, 21.1, 28.4]
    # sample2 = [5.6, 6.5, 6.6, 6.9, 9.2, 10.4, 10.6, 19.3]
    com_sample = sample1 + sample2
    com_sample.sort()
    sample1.sort()
    sample2.sort()

    for i in com_sample:
        if D < math.fabs(FX_obs(i, sample1) - FX_obs(i, sample2)):
            D = math.fabs(FX_obs(i, sample1) - FX_obs(i, sample2))
    # print(f"D_crit={1.36 * math.sqrt(1 / len(sample1) + 1 / len(sample2))}")
    return D


def FX_obs(x, L=[]):
    n = 0
    for i in L:
        if x < i:
            break
        n += 1
    return n / len(L)


def degree_distribution(G=nx.Graph()):
    degreeList = []

    for node in G.nodes:
        if G.degree[node] == 0:
            continue
        degreeList.append(G.degree[node])
    return degreeList


def hop_plot_distribution(G=nx.Graph()):
    hopPlotList = []
    dict = {}
    for v in G.nodes:
        dictTemp = nx.single_source_shortest_path_length(G, v)
        for k in dictTemp:
            if dictTemp[k] in dict:
                dict[dictTemp[k]] += 1
            else:
                dict[dictTemp[k]] = 1

    for k in dict:
        dict[k] /= 2
        if k == 0:
            continue
        for i in range(0, int(dict[k])):
            hopPlotList.append(k)

    return hopPlotList


def hop_plot_KS(G_hop_plot_List=[], GS_hop_plot_List=[]):
    G_hop_plot_Dict = {}

    GS_hop_plot_Dict = {}

    com_set = set()
    D = 0

    for hop_plot in G_hop_plot_List:
        if hop_plot in G_hop_plot_Dict:
            G_hop_plot_Dict[hop_plot] += 1
        else:
            G_hop_plot_Dict[hop_plot] = 1

    for hop_plot in GS_hop_plot_List:
        if hop_plot in GS_hop_plot_Dict:
            GS_hop_plot_Dict[hop_plot] += 1
        else:
            GS_hop_plot_Dict[hop_plot] = 1

    # 计算经验分布函数
    n_G = 0
    n_GS = 0
    # G_hop_plot_Dict.pop(0)
    # G_hop_plot_Dict.pop(0)

    for k in G_hop_plot_Dict:
        n_G += G_hop_plot_Dict[k]
        com_set.add(k)

    for k in GS_hop_plot_Dict:
        n_GS += GS_hop_plot_Dict[k]
        com_set.add(k)

    # print(dictG)
    # print(dictGS)
    for i in com_set:
        k_G = 0
        k_GS = 0

        for k in G_hop_plot_Dict:
            if k <= i:
                k_G += G_hop_plot_Dict[k]

        for k in GS_hop_plot_Dict:
            if k <= i:
                k_GS += GS_hop_plot_Dict[k]
        # print(i)
        # print(f"{k_G}\t{k_GS}\t{n_G}\t{n_GS}")
        if D < math.fabs(k_G / n_G - k_GS / n_GS):
            D = math.fabs(k_G / n_G - k_GS / n_GS)

    # print(f"D_crit={1.36 * math.sqrt(1 / n_G + 1 / n_GS)}")

    return D


def clustering_coefficient_distribution(G=nx.Graph()):
    ccList = []
    for v in G.nodes:
        if G.degree[v] <= 1:
            continue
        R_v = 0
        N_v = len(list(G.neighbors(v))) * (len(list(G.neighbors(v))) - 1)
        for i in G.neighbors(v):
            for k in G.neighbors(v):
                if G.has_edge(i, k):
                    R_v += 1
        ccList.append(R_v / N_v)
    ccList.sort()
    return ccList


def clustering_coefficient_KS(G_ccList=[], GS_ccList=[]):
    if len(G_ccList) == 0 or len(GS_ccList) == 0:
        return 1

    G_ccDict = {}

    GS_ccDict = {}

    D = 0
    comcc_set = set([])

    # 获取所有的cc值
    for cc in G_ccList:
        comcc_set.add(cc)
    for cc in GS_ccList:
        comcc_set.add(cc)

    # 统计cc值的个数
    for cc in G_ccList:
        if cc in G_ccDict:
            G_ccDict[cc] += 1
        else:
            G_ccDict[cc] = 1

    for cc in GS_ccList:
        if cc in GS_ccDict:
            GS_ccDict[cc] += 1
        else:
            GS_ccDict[cc] = 1

    # 计算经验分布函数
    G_n = len(G_ccList)
    GS_n = len(GS_ccList)
    # print(f"{G_n},{GS_n}")

    for cc in comcc_set:
        G_k = 0
        GS_k = 0
        for k in G_ccDict:
            if k <= cc:
                G_k += G_ccDict[k]

        for k in GS_ccDict:
            if k <= cc:
                GS_k += GS_ccDict[k]

        if D < math.fabs(G_k / G_n - GS_k / GS_n):
            D = math.fabs(G_k / G_n - GS_k / GS_n)

    # print(f"D_crit={1.36 * math.sqrt(1 / G_n + 1 / GS_n)}")
    return D


def k_core_distribution(G=nx.Graph()):
    k_core_list = []
    k_core_dict = nx.core_number(G)
    for k in k_core_dict:
        k_core_list.append(k_core_dict[k])
    return k_core_list


def k_core_KS(G_k_core_List=[], GS_k_core_List=[]):
    G_k_core_Dict = {}

    GS_k_core_Dict = {}

    D = 0
    com_k_core_set = set([])

    # 获取所有的k_core值
    for k_core in G_k_core_List:
        com_k_core_set.add(k_core)
    for k_core in GS_k_core_List:
        com_k_core_set.add(k_core)

    # 统计k_core值的个数
    for k_core in G_k_core_List:
        if k_core in G_k_core_Dict:
            G_k_core_Dict[k_core] += 1
        else:
            G_k_core_Dict[k_core] = 1

    for k_core in GS_k_core_List:
        if k_core in GS_k_core_Dict:
            GS_k_core_Dict[k_core] += 1
        else:
            GS_k_core_Dict[k_core] = 1

    # 计算经验分布函数
    G_n = len(G_k_core_List)
    GS_n = len(GS_k_core_List)
    for k_core in com_k_core_set:
        G_k = 0
        GS_k = 0
        for k in G_k_core_Dict:
            if k <= k_core:
                G_k += G_k_core_Dict[k]

        for k in GS_k_core_Dict:
            if k <= k_core:
                GS_k += GS_k_core_Dict[k]

        if D < math.fabs(G_k / G_n - GS_k / GS_n):
            D = math.fabs(G_k / G_n - GS_k / GS_n)

    # print(f"D_crit={1.36 * math.sqrt(1 / G_n + 1 / GS_n)}")
    return D
