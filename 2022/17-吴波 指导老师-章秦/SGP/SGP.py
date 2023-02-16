import networkx as nx

import GraphPartition as gp
import GraphSampling as gs
import Util as util
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

import KStest as ks
import KLtest as kl
import RW
import SGP
import SamplingMethods as sm


def SGP(G=nx.Graph(), P=0.2):
    GS = nx.Graph()  # 采样图
    GPartition = gp.graphPartition(G)
    subGraphs = util.getSubGraphs(GPartition)

    for k in subGraphs:
        subGS = gs.stratifiedSampling(subGraphs[k], G, P)

        for v in subGS.nodes:
            GS.add_node(v)

        for e in subGS.edges:
            GS.add_edge(e[0], e[1])

    for u in GS.nodes:
        for v in GS.nodes:
            if G.has_edge(u, v):
                GS.add_edge(u, v)

    return GS


def GPRW(G=nx.Graph(), P=0.2):
    GS = nx.Graph()  # 采样图
    GPartition = gp.graphPartition(G)
    subGraphs = util.getSubGraphs(GPartition)

    # print(len(subGraphs))
    for k in subGraphs:
        subGS = RW.RW(subGraphs[k], P)

        for v in subGS.nodes:
            GS.add_node(v)

        for e in subGS.edges:
            GS.add_edge(e[0], e[1])

    for u in GS.nodes:
        for v in GS.nodes:
            if G.has_edge(u, v):
                GS.add_edge(u, v)

    return GS


def compare_KS_KL(G=nx.Graph()):
    G_degree_distribution = ks.degree_distribution(G)
    G_hop_plot_distribution = ks.hop_plot_distribution(G)
    G_cc_distribution = ks.clustering_coefficient_distribution(G)
    G_k_core_distribution = ks.k_core_distribution(G)

    GS1_degree_ks = 0
    GS2_degree_ks = 0

    GS1_hop_plot_ks = 0
    GS2_hop_plot_ks = 0

    GS1_clustering_coefficient_ks = 0
    GS2_clustering_coefficient_ks = 0

    GS1_k_core_ks = 0
    GS2_k_core_ks = 0

    GS1_degree_kl = 0
    GS2_degree_kl = 0

    GS1_hop_plot_kl = 0
    GS2_hop_plot_kl = 0

    GS1_clustering_coefficient_kl = 0
    GS2_clustering_coefficient_kl = 0

    GS1_k_core_kl = 0
    GS2_k_core_kl = 0

    for i in range(0, 3):
        GS1 = SGP(G.copy())
        GS2 = GPRW(G.copy())

        GS1_degree_distribution = ks.degree_distribution(GS1)
        GS2_degree_distribution = ks.degree_distribution(GS2)

        GS1_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS1_degree_distribution)
        GS2_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS2_degree_distribution)

        GS1_hop_plot_distribution = ks.hop_plot_distribution(GS1)
        GS2_hop_plot_distribution = ks.hop_plot_distribution(GS2)

        GS1_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS1_hop_plot_distribution)
        GS2_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS2_hop_plot_distribution)

        GS1_cc_distribution = ks.clustering_coefficient_distribution(GS1)
        GS2_cc_distribution = ks.clustering_coefficient_distribution(GS2)

        GS1_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS1_cc_distribution)
        GS2_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS2_cc_distribution)

        GS1_k_core_distribution = ks.k_core_distribution(GS1)
        GS2_k_core_distribution = ks.k_core_distribution(GS2)

        GS1_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS1_k_core_distribution)
        GS2_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS2_k_core_distribution)

        GS1_degree_kl += kl.degree_KL(G, GS1)
        GS2_degree_kl += kl.degree_KL(G, GS2)

        GS1_hop_plot_kl += kl.hop_plot_KL(G_hop_plot_distribution, GS1_hop_plot_distribution)
        GS2_hop_plot_kl += kl.hop_plot_KL(G_hop_plot_distribution, GS2_hop_plot_distribution)

        GS1_clustering_coefficient_kl += kl.clustering_coefficient_KL(G_cc_distribution, GS1_cc_distribution)
        GS2_clustering_coefficient_kl += kl.clustering_coefficient_KL(G_cc_distribution, GS2_cc_distribution)

        GS1_k_core_kl += kl.k_core_KL(G_k_core_distribution, GS1_k_core_distribution)
        GS2_k_core_kl += kl.k_core_KL(G_k_core_distribution, GS2_k_core_distribution)

    print('degree_KS------------------------')
    print(round(GS1_degree_ks / 3, 6))
    print(round(GS2_degree_ks / 3, 6))
    print()
    print('hop_plot_KS----------------------')
    print(round(GS1_hop_plot_ks / 3, 6))
    print(round(GS2_hop_plot_ks / 3, 6))
    print()
    print('clustering_coefficient_KS----------')
    print(round(GS1_clustering_coefficient_ks / 3, 6))
    print(round(GS2_clustering_coefficient_ks / 3, 6))
    print()
    print('k_core_KS----------------------')
    print(round(GS1_k_core_ks / 3, 6))
    print(round(GS2_k_core_ks / 3, 6))
    print()
    print('degree_KL-----------------------------')
    print(round(GS1_degree_kl / 3, 6))
    print(round(GS2_degree_kl / 3, 6))
    print()
    print('hop_plot_KL-----------------------------')
    print(round(GS1_hop_plot_kl / 3, 6))
    print(round(GS2_hop_plot_kl / 3, 6))
    print()
    print('clustering_coefficient_KL-----------------------------')
    print(round(GS1_clustering_coefficient_kl / 3, 6))
    print(round(GS2_clustering_coefficient_kl / 3, 6))
    print()
    print('k_core_KL-----------------------------')
    print(round(GS1_k_core_kl / 3, 6))
    print(round(GS2_k_core_kl / 3, 6))
    print()


def drawCDF(G=nx.Graph(), GS1=nx.Graph(), GS2=nx.Graph()):
    fig, a = plt.subplots(2, 2)

    drawCDF_degree(a, G, GS1, GS2)
    drawCDF_hop_plot(a, G, GS1, GS2)
    drawCDF_clustering_coefficient(a, G, GS1, GS2)
    drawCDF_k_core(a, G, GS1, GS2)
    plt.subplots_adjust(hspace=0.45)  # 调整子图上下之间的距离
    plt.show()


def drawCDF_degree(a, G=nx.Graph(), GS1=nx.Graph(), GS2=nx.Graph()):
    # degreeCDF
    com_degree = set([])
    G_CDF_dict = {}  # 保存度为k的节点个数
    G_CDF_list = []  # 经验分布

    GS1_CDF_dict = {}
    GS1_CDF_list = []

    GS2_CDF_dict = {}
    GS2_CDF_list = []

    # 封装G的CDF
    for node in G.nodes:
        com_degree.add(G.degree[node])
        if G.degree[node] in G_CDF_dict:
            G_CDF_dict[G.degree[node]] += 1
        else:
            G_CDF_dict[G.degree[node]] = 1

    for node in GS1.nodes:
        com_degree.add(GS1.degree[node])
        if GS1.degree[node] in GS1_CDF_dict:
            GS1_CDF_dict[GS1.degree[node]] += 1
        else:
            GS1_CDF_dict[GS1.degree[node]] = 1

    for node in GS2.nodes:
        com_degree.add(GS2.degree[node])
        if GS2.degree[node] in GS2_CDF_dict:
            GS2_CDF_dict[GS2.degree[node]] += 1
        else:
            GS2_CDF_dict[GS2.degree[node]] = 1

    # 计算每个度的经验分布
    G_n = len(G.nodes)
    GS1_n = len(GS1.nodes)
    GS2_n = len(GS2.nodes)

    com_degree = list(com_degree)
    com_degree.sort()

    if 0 in G_CDF_dict:
        G_n -= G_CDF_dict[0]
    if 0 in GS1_CDF_dict:
        GS1_n -= GS1_CDF_dict[0]
    if 0 in GS2_CDF_dict:
        GS2_n -= GS2_CDF_dict[0]

    for degree in com_degree:
        G_k = 0
        GS1_k = 0
        GS2_k = 0

        for k in G_CDF_dict:
            if k == 0:
                continue
            if k <= degree:
                G_k += G_CDF_dict[k]

        for k in GS1_CDF_dict:
            if k == 0:
                continue
            if k <= degree:
                GS1_k += GS1_CDF_dict[k]

        for k in GS2_CDF_dict:
            if k == 0:
                continue
            if k <= degree:
                GS2_k += GS2_CDF_dict[k]

        G_CDF_list.append(G_k / G_n)
        GS1_CDF_list.append(GS1_k / GS1_n)
        GS2_CDF_list.append(GS2_k / GS2_n)

    # print(com_degree)
    # print(G_CDF_list)
    # print(GS1_CDF_list)

    a[0][0].plot(list(com_degree), G_CDF_list, '-r')
    a[0][0].plot(list(com_degree), GS1_CDF_list, '-g')
    a[0][0].plot(list(com_degree), GS2_CDF_list, '-b')

    a[0][0].legend(labels=('Original G', 'SGP', 'GPRW'), loc='lower right')
    a[0][0].set_xlabel('degree')
    a[0][0].set_ylabel('P(degree<x)')
    a[0][0].set_title('Degree CDF')


def drawCDF_hop_plot(a, G=nx.Graph(), GS1=nx.Graph(), GS2=nx.Graph()):
    # hop_plotCDF
    com_hop_plot = set([])
    G_CDF_dict = {}  # 保存hop_plot为k的节点个数
    G_CDF_list = []  # 经验分布

    GS1_CDF_dict = {}
    GS1_CDF_list = []

    GS2_CDF_dict = {}
    GS2_CDF_list = []

    # 获取初始图和采样图的hop_plot分布
    for v in G.nodes:
        dictTemp = nx.single_source_shortest_path_length(G, v)
        for k in dictTemp:
            if dictTemp[k] in G_CDF_dict:
                G_CDF_dict[dictTemp[k]] += 1
            else:
                G_CDF_dict[dictTemp[k]] = 1

    for v in GS1.nodes:
        dictTemp = nx.single_source_shortest_path_length(GS1, v)
        for k in dictTemp:
            if dictTemp[k] in GS1_CDF_dict:
                GS1_CDF_dict[dictTemp[k]] += 1
            else:
                GS1_CDF_dict[dictTemp[k]] = 1

    for v in GS2.nodes:
        dictTemp = nx.single_source_shortest_path_length(GS2, v)
        for k in dictTemp:
            if dictTemp[k] in GS2_CDF_dict:
                GS2_CDF_dict[dictTemp[k]] += 1
            else:
                GS2_CDF_dict[dictTemp[k]] = 1

    # 计算经验分布函数
    G_n = 0
    GS1_n = 0
    GS2_n = 0

    G_CDF_dict.pop(0)
    GS1_CDF_dict.pop(0)
    GS2_CDF_dict.pop(0)

    for k in G_CDF_dict:
        G_CDF_dict[k] /= 2
        G_n += G_CDF_dict[k]
        com_hop_plot.add(k)

    for k in GS1_CDF_dict:
        GS1_CDF_dict[k] /= 2
        GS1_n += GS1_CDF_dict[k]
        com_hop_plot.add(k)

    for k in GS2_CDF_dict:
        GS2_CDF_dict[k] /= 2
        GS2_n += GS2_CDF_dict[k]
        com_hop_plot.add(k)

    com_hop_plot = list(com_hop_plot)
    com_hop_plot.sort()

    for hop_plot in com_hop_plot:
        G_k = 0
        GS1_k = 0
        GS2_k = 0

        for k in G_CDF_dict:
            if k <= hop_plot:
                G_k += G_CDF_dict[k]

        for k in GS1_CDF_dict:
            if k <= hop_plot:
                GS1_k += GS1_CDF_dict[k]

        for k in GS2_CDF_dict:
            if k <= hop_plot:
                GS2_k += GS2_CDF_dict[k]

        G_CDF_list.append(G_k / G_n)
        GS1_CDF_list.append(GS1_k / GS1_n)
        GS2_CDF_list.append(GS2_k / GS2_n)

    a[0][1].plot(list(com_hop_plot), G_CDF_list, '.-r')
    a[0][1].plot(list(com_hop_plot), GS1_CDF_list, '.-g')
    a[0][1].plot(list(com_hop_plot), GS2_CDF_list, '.-b')

    a[0][1].legend(labels=('Original G', 'SGP', 'GPRW'), loc='lower right')
    a[0][1].set_xlabel('hop plot')
    a[0][1].set_ylabel('P(hop_plot<x)')
    a[0][1].set_title('Hop Plot CDF')


def drawCDF_clustering_coefficient(a, G=nx.Graph(), GS1=nx.Graph(), GS2=nx.Graph()):
    G_CDF_dict = {}  # 保存clustering_coefficient为cc的节点个数
    G_CDF_list = []  # 经验分布
    G_ccList = ks.clustering_coefficient_distribution(G)

    GS1_CDF_dict = {}
    GS1_CDF_list = []
    GS1_ccList = ks.clustering_coefficient_distribution(GS1)

    GS2_CDF_dict = {}
    GS2_CDF_list = []
    GS2_ccList = ks.clustering_coefficient_distribution(GS2)

    comcc_set = set([])

    # 获取所有的cc值
    for cc in G_ccList:
        comcc_set.add(cc)
    for cc in GS1_ccList:
        comcc_set.add(cc)
    for cc in GS2_ccList:
        comcc_set.add(cc)

    comcc_set = list(comcc_set)
    comcc_set.sort()

    # 统计cc值的个数
    for cc in G_ccList:
        if cc in G_CDF_dict:
            G_CDF_dict[cc] += 1
        else:
            G_CDF_dict[cc] = 1

    for cc in GS1_ccList:
        if cc in GS1_CDF_dict:
            GS1_CDF_dict[cc] += 1
        else:
            GS1_CDF_dict[cc] = 1

    for cc in GS2_ccList:
        if cc in GS2_CDF_dict:
            GS2_CDF_dict[cc] += 1
        else:
            GS2_CDF_dict[cc] = 1

    # 计算经验分布函数
    G_n = len(G_ccList)
    GS1_n = len(GS1_ccList)
    GS2_n = len(GS2_ccList)

    for cc in comcc_set:
        G_k = 0
        GS1_k = 0
        GS2_k = 0

        for k in G_CDF_dict:
            if k <= cc:
                G_k += G_CDF_dict[k]

        for k in GS1_CDF_dict:
            if k <= cc:
                GS1_k += GS1_CDF_dict[k]

        for k in GS2_CDF_dict:
            if k <= cc:
                GS2_k += GS2_CDF_dict[k]

        G_CDF_list.append(G_k / G_n)
        GS1_CDF_list.append(GS1_k / GS1_n)
        GS2_CDF_list.append(GS2_k / GS2_n)

    a[1][0].plot(list(comcc_set), G_CDF_list, '-r')
    a[1][0].plot(list(comcc_set), GS1_CDF_list, '-g')
    a[1][0].plot(list(comcc_set), GS2_CDF_list, '-b')
    a[1][0].legend(labels=('Original G', 'SGP', 'GPRW'), loc='lower right')
    a[1][0].set_xlabel('clustering coefficient')
    a[1][0].set_ylabel('P(clustering_coefficient<x)')
    a[1][0].set_title('Clustering Coefficient CDF')


def drawCDF_k_core(a, G=nx.Graph(), GS1=nx.Graph(), GS2=nx.Graph()):
    G_CDF_dict = {}  # 保存k_core为kcore的节点个数
    G_CDF_list = []  # 经验分布
    G_kcoreList = ks.k_core_distribution(G)

    GS1_CDF_dict = {}
    GS1_CDF_list = []
    GS1_kcoreList = ks.k_core_distribution(GS1)

    GS2_CDF_dict = {}
    GS2_CDF_list = []
    GS2_kcoreList = ks.k_core_distribution(GS2)

    comkcore_set = set([])

    # 获取所有的kcore值
    for kcore in G_kcoreList:
        comkcore_set.add(kcore)
    for kcore in GS1_kcoreList:
        comkcore_set.add(kcore)
    for kcore in GS2_kcoreList:
        comkcore_set.add(kcore)
    comkcore_set = list(comkcore_set)
    comkcore_set.sort()

    # 统计kcore值的个数
    for kcore in G_kcoreList:
        if kcore in G_CDF_dict:
            G_CDF_dict[kcore] += 1
        else:
            G_CDF_dict[kcore] = 1

    for kcore in GS1_kcoreList:
        if kcore in GS1_CDF_dict:
            GS1_CDF_dict[kcore] += 1
        else:
            GS1_CDF_dict[kcore] = 1

    for kcore in GS2_kcoreList:
        if kcore in GS2_CDF_dict:
            GS2_CDF_dict[kcore] += 1
        else:
            GS2_CDF_dict[kcore] = 1

    # 计算经验分布函数
    G_n = len(G_kcoreList)
    GS1_n = len(GS1_kcoreList)
    GS2_n = len(GS2_kcoreList)

    for kcore in comkcore_set:
        G_k = 0
        GS1_k = 0
        GS2_k = 0

        for k in G_CDF_dict:
            if k <= kcore:
                G_k += G_CDF_dict[k]

        for k in GS1_CDF_dict:
            if k <= kcore:
                GS1_k += GS1_CDF_dict[k]

        for k in GS2_CDF_dict:
            if k <= kcore:
                GS2_k += GS2_CDF_dict[k]

        G_CDF_list.append(G_k / G_n)
        GS1_CDF_list.append(GS1_k / GS1_n)
        GS2_CDF_list.append(GS2_k / GS2_n)

    a[1][1].plot(list(comkcore_set), G_CDF_list, '-r')
    a[1][1].plot(list(comkcore_set), GS1_CDF_list, '-g')
    a[1][1].plot(list(comkcore_set), GS2_CDF_list, '-b')

    a[1][1].legend(labels=('Original G', 'SGP', 'SGPW'), loc='lower right')
    a[1][1].set_xlabel('k core')
    a[1][1].set_ylabel('P(k_core<x)')
    a[1][1].set_title('K Core CDF')
