import networkx as nx
from scipy import stats
import random
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

import KStest as ks
import RW
import SGP
import SamplingMethods as sm


def drawGS(G=nx.Graph(), GPartition=nx.Graph(), GS=nx.Graph()):
    # 绘图
    fig = plt.figure("Sampling Graph", figsize=(8, 8))
    axgrid = fig.add_gridspec(6, 4)

    # 1.绘画原始图
    ax0 = fig.add_subplot(axgrid[0:2, :])
    pos = nx.spring_layout(G, seed=10396953)
    nx.draw_networkx_nodes(G, pos, ax=ax0, node_size=2, node_color='red')
    nx.draw_networkx_edges(G, pos, ax=ax0, alpha=0.2)
    # ax0.set_title("initial Graph")
    # ax0.set_axis_off()   # 去除网格线

    # 2.绘画图划分之后的图
    # 剔除GPartition度为0的孤立节点
    Draw_GPartition_nodes_list = []  # 剔除GPartition孤立节点后的结点列表
    connected_components = list(nx.connected_components(GPartition))

    for component in connected_components:
        if len(component) >= 2:
            Draw_GPartition_nodes_list += list(component)

    Draw_GPartition = GPartition.subgraph(Draw_GPartition_nodes_list)
    ax1 = fig.add_subplot(axgrid[2:4, :])
    pos = nx.spring_layout(Draw_GPartition, seed=10396953)
    nx.draw_networkx_nodes(Draw_GPartition, pos, ax=ax1, node_size=2, node_color='red')
    nx.draw_networkx_edges(Draw_GPartition, pos, ax=ax1, alpha=0.2)
    # ax1.set_title("GPatition")
    # ax1.set_axis_off()

    # 3.绘画采样图
    # 剔除GS度为0的孤立节点
    Draw_GS_nodes_list = []  # 剔除GS孤立节点后的结点列表
    connected_components = list(nx.connected_components(GS))

    for component in connected_components:
        if len(component) >= 2:
            Draw_GS_nodes_list += list(component)

    Draw_GS = GS.subgraph(Draw_GS_nodes_list)
    ax2 = fig.add_subplot(axgrid[4:, :])
    pos = nx.spring_layout(Draw_GS, seed=10396953)
    nx.draw_networkx_nodes(Draw_GS, pos, ax=ax2, node_size=2, node_color='red')
    nx.draw_networkx_edges(Draw_GS, pos, ax=ax2, alpha=0.2)
    # ax2.set_title("GS")

    plt.show()


def drawCDF(G=nx.Graph(), GS1=nx.Graph(), GS2=nx.Graph, GS3=nx.Graph(), GS4=nx.Graph(), GS5=nx.Graph(),
            GS6=nx.Graph()):
    fig, a = plt.subplots(2, 2)

    drawCDF_degree(a, G, GS1, GS2, GS3, GS4, GS5, GS6)
    drawCDF_hop_plot(a, G, GS1, GS2, GS3, GS4, GS5, GS6)
    drawCDF_clustering_coefficient(a, G, GS1, GS2, GS3, GS4, GS5, GS6)
    drawCDF_k_core(a, G, GS1, GS2, GS3, GS4, GS5, GS6)
    plt.subplots_adjust(hspace=0.45)  # 调整子图上下之间的距离
    plt.show()


def drawCDF_degree(a, G=nx.Graph(), GS1=nx.Graph(), GS2=nx.Graph(), GS3=nx.Graph(), GS4=nx.Graph(), GS5=nx.Graph(),
                   GS6=nx.Graph()):
    # degreeCDF
    com_degree = set([])
    G_CDF_dict = {}  # 保存度为k的节点个数
    G_CDF_list = []  # 经验分布

    GS1_CDF_dict = {}
    GS1_CDF_list = []

    GS2_CDF_dict = {}
    GS2_CDF_list = []

    GS3_CDF_dict = {}
    GS3_CDF_list = []

    GS4_CDF_dict = {}
    GS4_CDF_list = []

    GS5_CDF_dict = {}
    GS5_CDF_list = []

    GS6_CDF_dict = {}
    GS6_CDF_list = []

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

    for node in GS3.nodes:
        com_degree.add(GS3.degree[node])
        if GS3.degree[node] in GS3_CDF_dict:
            GS3_CDF_dict[GS3.degree[node]] += 1
        else:
            GS3_CDF_dict[GS3.degree[node]] = 1

    for node in GS4.nodes:
        com_degree.add(GS4.degree[node])
        if GS4.degree[node] in GS4_CDF_dict:
            GS4_CDF_dict[GS4.degree[node]] += 1
        else:
            GS4_CDF_dict[GS4.degree[node]] = 1

    for node in GS5.nodes:
        com_degree.add(GS5.degree[node])
        if GS5.degree[node] in GS5_CDF_dict:
            GS5_CDF_dict[GS5.degree[node]] += 1
        else:
            GS5_CDF_dict[GS5.degree[node]] = 1

    for node in GS6.nodes:
        com_degree.add(GS6.degree[node])
        if GS6.degree[node] in GS6_CDF_dict:
            GS6_CDF_dict[GS6.degree[node]] += 1
        else:
            GS6_CDF_dict[GS6.degree[node]] = 1

    # 计算每个度的经验分布
    G_n = len(G.nodes)
    GS1_n = len(GS1.nodes)
    GS2_n = len(GS2.nodes)
    GS3_n = len(GS3.nodes)
    GS4_n = len(GS4.nodes)
    GS5_n = len(GS5.nodes)
    GS6_n = len(GS6.nodes)

    com_degree = list(com_degree)
    com_degree.sort()

    if 0 in G_CDF_dict:
        G_n -= G_CDF_dict[0]
    if 0 in GS1_CDF_dict:
        GS1_n -= GS1_CDF_dict[0]
    if 0 in GS2_CDF_dict:
        GS2_n -= GS2_CDF_dict[0]
    if 0 in GS3_CDF_dict:
        GS3_n -= GS3_CDF_dict[0]
    if 0 in GS4_CDF_dict:
        GS4_n -= GS4_CDF_dict[0]
    if 0 in GS5_CDF_dict:
        GS5_n -= GS5_CDF_dict[0]
    if 0 in GS6_CDF_dict:
        GS6_n -= GS6_CDF_dict[0]

    for degree in com_degree:
        G_k = 0
        GS1_k = 0
        GS2_k = 0
        GS3_k = 0
        GS4_k = 0
        GS5_k = 0
        GS6_k = 0

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

        for k in GS3_CDF_dict:
            if k == 0:
                continue
            if k <= degree:
                GS3_k += GS3_CDF_dict[k]

        for k in GS4_CDF_dict:
            if k == 0:
                continue
            if k <= degree:
                GS4_k += GS4_CDF_dict[k]

        for k in GS5_CDF_dict:
            if k == 0:
                continue
            if k <= degree:
                GS5_k += GS5_CDF_dict[k]

        for k in GS6_CDF_dict:
            if k == 0:
                continue
            if k <= degree:
                GS6_k += GS6_CDF_dict[k]

        G_CDF_list.append(G_k / G_n)
        GS1_CDF_list.append(GS1_k / GS1_n)
        GS2_CDF_list.append(GS2_k / GS2_n)
        GS3_CDF_list.append(GS3_k / GS3_n)
        GS4_CDF_list.append(GS4_k / GS4_n)
        GS5_CDF_list.append(GS5_k / GS5_n)
        GS6_CDF_list.append(GS6_k / GS6_n)

    # print(com_degree)
    # print(G_CDF_list)
    # print(GS1_CDF_list)

    a[0][0].plot(list(com_degree), G_CDF_list, '-r')
    a[0][0].plot(list(com_degree), GS1_CDF_list, '-g')
    a[0][0].plot(list(com_degree), GS2_CDF_list, '-b')
    a[0][0].plot(list(com_degree), GS3_CDF_list, '-y')
    a[0][0].plot(list(com_degree), GS4_CDF_list, '-k')
    a[0][0].plot(list(com_degree), GS5_CDF_list, '-')
    a[0][0].plot(list(com_degree), GS6_CDF_list, '-')
    a[0][0].legend(labels=('G', 'MHRW', 'MHRJ', 'RW', 'RJ', 'FFS', 'SGP'), loc='lower right')
    a[0][0].set_xlabel('degree')
    a[0][0].set_ylabel('P(degree<x)')
    a[0][0].set_title('Degree CDF')


def drawCDF_hop_plot(a, G=nx.Graph(), GS1=nx.Graph(), GS2=nx.Graph(), GS3=nx.Graph(), GS4=nx.Graph(), GS5=nx.Graph(),
                     GS6=nx.Graph()):
    # hop_plotCDF
    com_hop_plot = set([])
    G_CDF_dict = {}  # 保存hop_plot为k的节点个数
    G_CDF_list = []  # 经验分布

    GS1_CDF_dict = {}
    GS1_CDF_list = []

    GS2_CDF_dict = {}
    GS2_CDF_list = []

    GS3_CDF_dict = {}
    GS3_CDF_list = []

    GS4_CDF_dict = {}
    GS4_CDF_list = []

    GS5_CDF_dict = {}
    GS5_CDF_list = []

    GS6_CDF_dict = {}
    GS6_CDF_list = []

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

    for v in GS3.nodes:
        dictTemp = nx.single_source_shortest_path_length(GS3, v)
        for k in dictTemp:
            if dictTemp[k] in GS3_CDF_dict:
                GS3_CDF_dict[dictTemp[k]] += 1
            else:
                GS3_CDF_dict[dictTemp[k]] = 1

    for v in GS4.nodes:
        dictTemp = nx.single_source_shortest_path_length(GS4, v)
        for k in dictTemp:
            if dictTemp[k] in GS4_CDF_dict:
                GS4_CDF_dict[dictTemp[k]] += 1
            else:
                GS4_CDF_dict[dictTemp[k]] = 1

    for v in GS5.nodes:
        dictTemp = nx.single_source_shortest_path_length(GS5, v)
        for k in dictTemp:
            if dictTemp[k] in GS5_CDF_dict:
                GS5_CDF_dict[dictTemp[k]] += 1
            else:
                GS5_CDF_dict[dictTemp[k]] = 1

    for v in GS6.nodes:
        dictTemp = nx.single_source_shortest_path_length(GS6, v)
        for k in dictTemp:
            if dictTemp[k] in GS6_CDF_dict:
                GS6_CDF_dict[dictTemp[k]] += 1
            else:
                GS6_CDF_dict[dictTemp[k]] = 1

    # 计算经验分布函数
    G_n = 0
    GS1_n = 0
    GS2_n = 0
    GS3_n = 0
    GS4_n = 0
    GS5_n = 0
    GS6_n = 0

    G_CDF_dict.pop(0)
    GS1_CDF_dict.pop(0)
    GS2_CDF_dict.pop(0)
    GS3_CDF_dict.pop(0)
    GS4_CDF_dict.pop(0)
    GS5_CDF_dict.pop(0)
    GS6_CDF_dict.pop(0)

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

    for k in GS3_CDF_dict:
        GS3_CDF_dict[k] /= 2
        GS3_n += GS3_CDF_dict[k]
        com_hop_plot.add(k)

    for k in GS4_CDF_dict:
        GS4_CDF_dict[k] /= 2
        GS4_n += GS4_CDF_dict[k]
        com_hop_plot.add(k)

    for k in GS5_CDF_dict:
        GS5_CDF_dict[k] /= 2
        GS5_n += GS5_CDF_dict[k]
        com_hop_plot.add(k)

    for k in GS6_CDF_dict:
        GS6_CDF_dict[k] /= 2
        GS6_n += GS6_CDF_dict[k]
        com_hop_plot.add(k)

    com_hop_plot = list(com_hop_plot)
    com_hop_plot.sort()

    for hop_plot in com_hop_plot:
        G_k = 0
        GS1_k = 0
        GS2_k = 0
        GS3_k = 0
        GS4_k = 0
        GS5_k = 0
        GS6_k = 0

        for k in G_CDF_dict:
            if k <= hop_plot:
                G_k += G_CDF_dict[k]

        for k in GS1_CDF_dict:
            if k <= hop_plot:
                GS1_k += GS1_CDF_dict[k]

        for k in GS2_CDF_dict:
            if k <= hop_plot:
                GS2_k += GS2_CDF_dict[k]

        for k in GS3_CDF_dict:
            if k <= hop_plot:
                GS3_k += GS3_CDF_dict[k]

        for k in GS4_CDF_dict:
            if k <= hop_plot:
                GS4_k += GS4_CDF_dict[k]

        for k in GS5_CDF_dict:
            if k <= hop_plot:
                GS5_k += GS5_CDF_dict[k]

        for k in GS6_CDF_dict:
            if k <= hop_plot:
                GS6_k += GS6_CDF_dict[k]

        G_CDF_list.append(G_k / G_n)
        GS1_CDF_list.append(GS1_k / GS1_n)
        GS2_CDF_list.append(GS2_k / GS2_n)
        GS3_CDF_list.append(GS3_k / GS3_n)
        GS4_CDF_list.append(GS4_k / GS4_n)
        GS5_CDF_list.append(GS5_k / GS5_n)
        GS6_CDF_list.append(GS6_k / GS6_n)

    # print(com_hop_plot)
    # print(G_CDF_list)
    # print(GS1_CDF_list)
    a[0][1].plot(list(com_hop_plot), G_CDF_list, '.-r')
    a[0][1].plot(list(com_hop_plot), GS1_CDF_list, '.-g')
    a[0][1].plot(list(com_hop_plot), GS2_CDF_list, '.-b')
    a[0][1].plot(list(com_hop_plot), GS3_CDF_list, '.-y')
    a[0][1].plot(list(com_hop_plot), GS4_CDF_list, '.-k')
    a[0][1].plot(list(com_hop_plot), GS5_CDF_list, '.-')
    a[0][1].plot(list(com_hop_plot), GS6_CDF_list, '.-')
    a[0][1].legend(labels=('G', 'MHRW', 'MHRJ', 'RW', 'RJ', 'FFS', 'SGP'), loc='lower right')
    a[0][1].set_xlabel('hop plot')
    a[0][1].set_ylabel('P(hop_plot<x)')
    a[0][1].set_title('Hop Plot CDF')


def drawCDF_clustering_coefficient(a, G=nx.Graph(), GS1=nx.Graph(), GS2=nx.Graph(), GS3=nx.Graph(), GS4=nx.Graph(),
                                   GS5=nx.Graph(),
                                   GS6=nx.Graph()):
    G_CDF_dict = {}  # 保存clustering_coefficient为cc的节点个数
    G_CDF_list = []  # 经验分布
    G_ccList = ks.clustering_coefficient_distribution(G)

    GS1_CDF_dict = {}
    GS1_CDF_list = []
    GS1_ccList = ks.clustering_coefficient_distribution(GS1)

    GS2_CDF_dict = {}
    GS2_CDF_list = []
    GS2_ccList = ks.clustering_coefficient_distribution(GS2)

    GS3_CDF_dict = {}
    GS3_CDF_list = []
    GS3_ccList = ks.clustering_coefficient_distribution(GS3)

    GS4_CDF_dict = {}
    GS4_CDF_list = []
    GS4_ccList = ks.clustering_coefficient_distribution(GS4)

    GS5_CDF_dict = {}
    GS5_CDF_list = []
    GS5_ccList = ks.clustering_coefficient_distribution(GS5)

    GS6_CDF_dict = {}
    GS6_CDF_list = []
    GS6_ccList = ks.clustering_coefficient_distribution(GS6)

    comcc_set = set([])

    # 获取所有的cc值
    for cc in G_ccList:
        comcc_set.add(cc)
    for cc in GS1_ccList:
        comcc_set.add(cc)
    for cc in GS2_ccList:
        comcc_set.add(cc)
    for cc in GS3_ccList:
        comcc_set.add(cc)
    for cc in GS4_ccList:
        comcc_set.add(cc)
    for cc in GS5_ccList:
        comcc_set.add(cc)
    for cc in GS6_ccList:
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

    for cc in GS3_ccList:
        if cc in GS3_CDF_dict:
            GS3_CDF_dict[cc] += 1
        else:
            GS3_CDF_dict[cc] = 1

    for cc in GS4_ccList:
        if cc in GS4_CDF_dict:
            GS4_CDF_dict[cc] += 1
        else:
            GS4_CDF_dict[cc] = 1

    for cc in GS5_ccList:
        if cc in GS5_CDF_dict:
            GS5_CDF_dict[cc] += 1
        else:
            GS5_CDF_dict[cc] = 1

    for cc in GS6_ccList:
        if cc in GS6_CDF_dict:
            GS6_CDF_dict[cc] += 1
        else:
            GS6_CDF_dict[cc] = 1

    # 计算经验分布函数
    G_n = len(G_ccList)
    GS1_n = len(GS1_ccList)
    GS2_n = len(GS2_ccList)
    GS3_n = len(GS3_ccList)
    GS4_n = len(GS4_ccList)
    GS5_n = len(GS5_ccList)
    GS6_n = len(GS6_ccList)

    for cc in comcc_set:
        G_k = 0
        GS1_k = 0
        GS2_k = 0
        GS3_k = 0
        GS4_k = 0
        GS5_k = 0
        GS6_k = 0

        for k in G_CDF_dict:
            if k <= cc:
                G_k += G_CDF_dict[k]

        for k in GS1_CDF_dict:
            if k <= cc:
                GS1_k += GS1_CDF_dict[k]

        for k in GS2_CDF_dict:
            if k <= cc:
                GS2_k += GS2_CDF_dict[k]

        for k in GS3_CDF_dict:
            if k <= cc:
                GS3_k += GS3_CDF_dict[k]

        for k in GS4_CDF_dict:
            if k <= cc:
                GS4_k += GS4_CDF_dict[k]

        for k in GS5_CDF_dict:
            if k <= cc:
                GS5_k += GS5_CDF_dict[k]

        for k in GS6_CDF_dict:
            if k <= cc:
                GS6_k += GS6_CDF_dict[k]

        G_CDF_list.append(G_k / G_n)
        GS1_CDF_list.append(GS1_k / GS1_n)
        GS2_CDF_list.append(GS2_k / GS2_n)
        GS3_CDF_list.append(GS3_k / GS3_n)
        GS4_CDF_list.append(GS4_k / GS4_n)
        GS5_CDF_list.append(GS5_k / GS5_n)
        GS6_CDF_list.append(GS6_k / GS6_n)

    a[1][0].plot(list(comcc_set), G_CDF_list, '-r')
    a[1][0].plot(list(comcc_set), GS1_CDF_list, '-g')
    a[1][0].plot(list(comcc_set), GS2_CDF_list, '-b')
    a[1][0].plot(list(comcc_set), GS3_CDF_list, '-y')
    a[1][0].plot(list(comcc_set), GS4_CDF_list, '-k')
    a[1][0].plot(list(comcc_set), GS5_CDF_list, '-')
    a[1][0].plot(list(comcc_set), GS6_CDF_list, '-')
    a[1][0].legend(labels=('G', 'MHRW', 'MHRJ', 'RW', 'RJ', 'FFS', 'SGP'), loc='lower right')
    a[1][0].set_xlabel('clustering coefficient')
    a[1][0].set_ylabel('P(clustering_coefficient<x)')
    a[1][0].set_title('Clustering Coefficient CDF')


def drawCDF_k_core(a, G=nx.Graph(), GS1=nx.Graph(), GS2=nx.Graph(), GS3=nx.Graph(), GS4=nx.Graph(), GS5=nx.Graph(),
                   GS6=nx.Graph()):
    G_CDF_dict = {}  # 保存k_core为kcore的节点个数
    G_CDF_list = []  # 经验分布
    G_kcoreList = ks.k_core_distribution(G)

    GS1_CDF_dict = {}
    GS1_CDF_list = []
    GS1_kcoreList = ks.k_core_distribution(GS1)

    GS2_CDF_dict = {}
    GS2_CDF_list = []
    GS2_kcoreList = ks.k_core_distribution(GS2)

    GS3_CDF_dict = {}
    GS3_CDF_list = []
    GS3_kcoreList = ks.k_core_distribution(GS3)

    GS4_CDF_dict = {}
    GS4_CDF_list = []
    GS4_kcoreList = ks.k_core_distribution(GS4)

    GS5_CDF_dict = {}
    GS5_CDF_list = []
    GS5_kcoreList = ks.k_core_distribution(GS5)

    GS6_CDF_dict = {}
    GS6_CDF_list = []
    GS6_kcoreList = ks.k_core_distribution(GS6)

    comkcore_set = set([])

    # 获取所有的kcore值
    for kcore in G_kcoreList:
        comkcore_set.add(kcore)
    for kcore in GS1_kcoreList:
        comkcore_set.add(kcore)
    for kcore in GS2_kcoreList:
        comkcore_set.add(kcore)
    for kcore in GS3_kcoreList:
        comkcore_set.add(kcore)
    for kcore in GS4_kcoreList:
        comkcore_set.add(kcore)
    for kcore in GS5_kcoreList:
        comkcore_set.add(kcore)
    for kcore in GS6_kcoreList:
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

    for kcore in GS3_kcoreList:
        if kcore in GS3_CDF_dict:
            GS3_CDF_dict[kcore] += 1
        else:
            GS3_CDF_dict[kcore] = 1

    for kcore in GS4_kcoreList:
        if kcore in GS4_CDF_dict:
            GS4_CDF_dict[kcore] += 1
        else:
            GS4_CDF_dict[kcore] = 1

    for kcore in GS5_kcoreList:
        if kcore in GS5_CDF_dict:
            GS5_CDF_dict[kcore] += 1
        else:
            GS5_CDF_dict[kcore] = 1

    for kcore in GS6_kcoreList:
        if kcore in GS6_CDF_dict:
            GS6_CDF_dict[kcore] += 1
        else:
            GS6_CDF_dict[kcore] = 1

    # 计算经验分布函数
    G_n = len(G_kcoreList)
    GS1_n = len(GS1_kcoreList)
    GS2_n = len(GS2_kcoreList)
    GS3_n = len(GS3_kcoreList)
    GS4_n = len(GS4_kcoreList)
    GS5_n = len(GS5_kcoreList)
    GS6_n = len(GS6_kcoreList)

    for kcore in comkcore_set:
        G_k = 0
        GS1_k = 0
        GS2_k = 0
        GS3_k = 0
        GS4_k = 0
        GS5_k = 0
        GS6_k = 0

        for k in G_CDF_dict:
            if k <= kcore:
                G_k += G_CDF_dict[k]

        for k in GS1_CDF_dict:
            if k <= kcore:
                GS1_k += GS1_CDF_dict[k]

        for k in GS2_CDF_dict:
            if k <= kcore:
                GS2_k += GS2_CDF_dict[k]

        for k in GS3_CDF_dict:
            if k <= kcore:
                GS3_k += GS3_CDF_dict[k]

        for k in GS4_CDF_dict:
            if k <= kcore:
                GS4_k += GS4_CDF_dict[k]

        for k in GS5_CDF_dict:
            if k <= kcore:
                GS5_k += GS5_CDF_dict[k]

        for k in GS6_CDF_dict:
            if k <= kcore:
                GS6_k += GS6_CDF_dict[k]

        G_CDF_list.append(G_k / G_n)
        GS1_CDF_list.append(GS1_k / GS1_n)
        GS2_CDF_list.append(GS2_k / GS2_n)
        GS3_CDF_list.append(GS3_k / GS3_n)
        GS4_CDF_list.append(GS4_k / GS4_n)
        GS5_CDF_list.append(GS5_k / GS5_n)
        GS6_CDF_list.append(GS6_k / GS6_n)

    a[1][1].plot(list(comkcore_set), G_CDF_list, '.-r')
    a[1][1].plot(list(comkcore_set), GS1_CDF_list, '.-g')
    a[1][1].plot(list(comkcore_set), GS2_CDF_list, '.-b')
    a[1][1].plot(list(comkcore_set), GS3_CDF_list, '.-y')
    a[1][1].plot(list(comkcore_set), GS4_CDF_list, '.-k')
    a[1][1].plot(list(comkcore_set), GS5_CDF_list, '.-')
    a[1][1].plot(list(comkcore_set), GS6_CDF_list, '.-')
    a[1][1].legend(labels=('G', 'MHRW', 'MHRJ', 'RW', 'RJ', 'FFS', 'SGP'), loc='lower right')
    a[1][1].set_xlabel('k core')
    a[1][1].set_ylabel('P(k_core<x)')
    a[1][1].set_title('K Core CDF')


def drawAvgKS(G=nx.Graph()):
    fig, a = plt.subplots(2, 2)

    sampleProportion = [0.1, 0.15, 0.2, 0.25, 0.3]
    GS1_degree_ks_list = []
    GS2_degree_ks_list = []
    GS3_degree_ks_list = []
    GS4_degree_ks_list = []
    GS5_degree_ks_list = []
    GS6_degree_ks_list = []

    GS1_hop_plot_ks_list = []
    GS2_hop_plot_ks_list = []
    GS3_hop_plot_ks_list = []
    GS4_hop_plot_ks_list = []
    GS5_hop_plot_ks_list = []
    GS6_hop_plot_ks_list = []

    GS1_clustering_coefficient_ks_list = []
    GS2_clustering_coefficient_ks_list = []
    GS3_clustering_coefficient_ks_list = []
    GS4_clustering_coefficient_ks_list = []
    GS5_clustering_coefficient_ks_list = []
    GS6_clustering_coefficient_ks_list = []

    GS1_k_core_ks_list = []
    GS2_k_core_ks_list = []
    GS3_k_core_ks_list = []
    GS4_k_core_ks_list = []
    GS5_k_core_ks_list = []
    GS6_k_core_ks_list = []

    G_degree_distribution = ks.degree_distribution(G)
    G_hop_plot_distribution = ks.hop_plot_distribution(G)
    G_cc_distribution = ks.clustering_coefficient_distribution(G)
    G_k_core_distribution = ks.k_core_distribution(G)

    for P in sampleProportion:
        GS1_degree_ks = 0
        GS2_degree_ks = 0
        GS3_degree_ks = 0
        GS4_degree_ks = 0
        GS5_degree_ks = 0
        GS6_degree_ks = 0

        GS1_hop_plot_ks = 0
        GS2_hop_plot_ks = 0
        GS3_hop_plot_ks = 0
        GS4_hop_plot_ks = 0
        GS5_hop_plot_ks = 0
        GS6_hop_plot_ks = 0

        GS1_clustering_coefficient_ks = 0
        GS2_clustering_coefficient_ks = 0
        GS3_clustering_coefficient_ks = 0
        GS4_clustering_coefficient_ks = 0
        GS5_clustering_coefficient_ks = 0
        GS6_clustering_coefficient_ks = 0

        GS1_k_core_ks = 0
        GS2_k_core_ks = 0
        GS3_k_core_ks = 0
        GS4_k_core_ks = 0
        GS5_k_core_ks = 0
        GS6_k_core_ks = 0

        for i in range(0, 5):
            GS1 = RW.Meropolis_Hastings_RW(G.copy(), P)
            GS2 = RW.Meropolis_Hastings_RJ(G.copy(), P)
            GS3 = RW.RW(G.copy(), P)
            GS4 = RW.RJ(G.copy(), P)
            GS5 = sm.FFS(G.copy(), P)
            GS6 = SGP.SGP(G.copy(), P)

            GS1_degree_distribution = ks.degree_distribution(GS1)
            GS2_degree_distribution = ks.degree_distribution(GS2)
            GS3_degree_distribution = ks.degree_distribution(GS3)
            GS4_degree_distribution = ks.degree_distribution(GS4)
            GS5_degree_distribution = ks.degree_distribution(GS5)
            GS6_degree_distribution = ks.degree_distribution(GS6)

            GS1_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS1_degree_distribution)
            GS2_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS2_degree_distribution)
            GS3_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS3_degree_distribution)
            GS4_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS4_degree_distribution)
            GS5_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS5_degree_distribution)
            GS6_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS6_degree_distribution)

            GS1_hop_plot_distribution = ks.hop_plot_distribution(GS1)
            GS2_hop_plot_distribution = ks.hop_plot_distribution(GS2)
            GS3_hop_plot_distribution = ks.hop_plot_distribution(GS3)
            GS4_hop_plot_distribution = ks.hop_plot_distribution(GS4)
            GS5_hop_plot_distribution = ks.hop_plot_distribution(GS5)
            GS6_hop_plot_distribution = ks.hop_plot_distribution(GS6)

            GS1_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS1_hop_plot_distribution)
            GS2_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS2_hop_plot_distribution)
            GS3_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS3_hop_plot_distribution)
            GS4_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS4_hop_plot_distribution)
            GS5_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS5_hop_plot_distribution)
            GS6_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS6_hop_plot_distribution)

            GS1_cc_distribution = ks.clustering_coefficient_distribution(GS1)
            GS2_cc_distribution = ks.clustering_coefficient_distribution(GS2)
            GS3_cc_distribution = ks.clustering_coefficient_distribution(GS3)
            GS4_cc_distribution = ks.clustering_coefficient_distribution(GS4)
            GS5_cc_distribution = ks.clustering_coefficient_distribution(GS5)
            GS6_cc_distribution = ks.clustering_coefficient_distribution(GS6)

            GS1_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS1_cc_distribution)
            GS2_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS2_cc_distribution)
            GS3_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS3_cc_distribution)
            GS4_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS4_cc_distribution)
            GS5_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS5_cc_distribution)
            GS6_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS6_cc_distribution)

            GS1_k_core_distribution = ks.k_core_distribution(GS1)
            GS2_k_core_distribution = ks.k_core_distribution(GS2)
            GS3_k_core_distribution = ks.k_core_distribution(GS3)
            GS4_k_core_distribution = ks.k_core_distribution(GS4)
            GS5_k_core_distribution = ks.k_core_distribution(GS5)
            GS6_k_core_distribution = ks.k_core_distribution(GS6)

            GS1_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS1_k_core_distribution)
            GS2_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS2_k_core_distribution)
            GS3_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS3_k_core_distribution)
            GS4_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS4_k_core_distribution)
            GS5_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS5_k_core_distribution)
            GS6_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS6_k_core_distribution)

        GS1_degree_ks_list.append(GS1_degree_ks / 5)
        GS2_degree_ks_list.append(GS2_degree_ks / 5)
        GS3_degree_ks_list.append(GS3_degree_ks / 5)
        GS4_degree_ks_list.append(GS4_degree_ks / 5)
        GS5_degree_ks_list.append(GS5_degree_ks / 5)
        GS6_degree_ks_list.append(GS6_degree_ks / 5)

        GS1_hop_plot_ks_list.append(GS1_hop_plot_ks / 5)
        GS2_hop_plot_ks_list.append(GS2_hop_plot_ks / 5)
        GS3_hop_plot_ks_list.append(GS3_hop_plot_ks / 5)
        GS4_hop_plot_ks_list.append(GS4_hop_plot_ks / 5)
        GS5_hop_plot_ks_list.append(GS5_hop_plot_ks / 5)
        GS6_hop_plot_ks_list.append(GS6_hop_plot_ks / 5)

        GS1_clustering_coefficient_ks_list.append(GS1_clustering_coefficient_ks / 5)
        GS2_clustering_coefficient_ks_list.append(GS2_clustering_coefficient_ks / 5)
        GS3_clustering_coefficient_ks_list.append(GS3_clustering_coefficient_ks / 5)
        GS4_clustering_coefficient_ks_list.append(GS4_clustering_coefficient_ks / 5)
        GS5_clustering_coefficient_ks_list.append(GS5_clustering_coefficient_ks / 5)
        GS6_clustering_coefficient_ks_list.append(GS6_clustering_coefficient_ks / 5)

        GS1_k_core_ks_list.append(GS1_k_core_ks / 5)
        GS2_k_core_ks_list.append(GS2_k_core_ks / 5)
        GS3_k_core_ks_list.append(GS3_k_core_ks / 5)
        GS4_k_core_ks_list.append(GS4_k_core_ks / 5)
        GS5_k_core_ks_list.append(GS5_k_core_ks / 5)
        GS6_k_core_ks_list.append(GS6_k_core_ks / 5)

    a[0][0].plot(sampleProportion, GS1_degree_ks_list, '.-r')
    a[0][0].plot(sampleProportion, GS2_degree_ks_list, '.-g')
    a[0][0].plot(sampleProportion, GS3_degree_ks_list, '.-b')
    a[0][0].plot(sampleProportion, GS4_degree_ks_list, '.-y')
    a[0][0].plot(sampleProportion, GS5_degree_ks_list, '.-k')
    a[0][0].plot(sampleProportion, GS6_degree_ks_list, '.-')
    a[0][0].legend(labels=('MHRW', 'MHRJ', 'RW', 'RJ', 'FFS', 'SGP'), loc='lower right')
    a[0][0].set_xticks(sampleProportion)
    a[0][0].set_xlabel('Sampling Fraction')
    a[0][0].set_yticks(np.arange(0, 1, 0.1))
    a[0][0].set_ylabel('Average KS Distance')
    a[0][0].set_title('Degree')

    a[0][1].plot(sampleProportion, GS1_hop_plot_ks_list, '.-r')
    a[0][1].plot(sampleProportion, GS2_hop_plot_ks_list, '.-g')
    a[0][1].plot(sampleProportion, GS3_hop_plot_ks_list, '.-b')
    a[0][1].plot(sampleProportion, GS4_hop_plot_ks_list, '.-y')
    a[0][1].plot(sampleProportion, GS5_hop_plot_ks_list, '.-k')
    a[0][1].plot(sampleProportion, GS6_hop_plot_ks_list, '.-')
    a[0][1].legend(labels=('MHRW', 'MHRJ', 'RW', 'RJ', 'FFS', 'SGP'), loc='lower right')
    a[0][1].set_xticks(sampleProportion)
    a[0][1].set_xlabel('Sampling Fraction')
    a[0][1].set_yticks(np.arange(0, 1, 0.1))
    a[0][1].set_ylabel('Average KS Distance')
    a[0][1].set_title('Hop Plot')

    a[1][0].plot(sampleProportion, GS1_clustering_coefficient_ks_list, '.-r')
    a[1][0].plot(sampleProportion, GS2_clustering_coefficient_ks_list, '.-g')
    a[1][0].plot(sampleProportion, GS3_clustering_coefficient_ks_list, '.-b')
    a[1][0].plot(sampleProportion, GS4_clustering_coefficient_ks_list, '.-y')
    a[1][0].plot(sampleProportion, GS5_clustering_coefficient_ks_list, '.-k')
    a[1][0].plot(sampleProportion, GS6_clustering_coefficient_ks_list, '.-')
    a[1][0].legend(labels=('MHRW', 'MHRJ', 'RW', 'RJ', 'FFS', 'SGP'), loc='lower right')
    a[1][0].set_xticks(sampleProportion)
    a[1][0].set_xlabel('Sampling Fraction')
    a[1][0].set_yticks(np.arange(0, 1, 0.1))
    a[1][0].set_ylabel('Average KS Distance')
    a[1][0].set_title('Clustering Coefficient')

    a[1][1].plot(sampleProportion, GS1_k_core_ks_list, '.-r')
    a[1][1].plot(sampleProportion, GS2_k_core_ks_list, '.-g')
    a[1][1].plot(sampleProportion, GS3_k_core_ks_list, '.-b')
    a[1][1].plot(sampleProportion, GS4_k_core_ks_list, '.-y')
    a[1][1].plot(sampleProportion, GS5_k_core_ks_list, '.-k')
    a[1][1].plot(sampleProportion, GS6_k_core_ks_list, '.-')
    a[1][1].legend(labels=('MHRW', 'MHRJ', 'RW', 'RJ', 'FFS', 'SGP'), loc='lower right')
    a[1][1].set_xticks(sampleProportion)
    a[1][1].set_xlabel('Sampling Fraction')
    a[1][1].set_yticks(np.arange(0, 1.1, 0.1))
    a[1][1].set_ylabel('Average KS Distance')
    a[1][1].set_title('K Core')

    plt.subplots_adjust(hspace=0.45)  # 调整子图上下之间的距离
    plt.show()
