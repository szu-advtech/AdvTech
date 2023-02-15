import networkx as nx
import math

import KStest as ks


def degree_KL(G=nx.Graph(), GS=nx.Graph()):
    G_degree_distribution = ks.degree_distribution(G)  # 度分布
    G_degree_dict = {}  # 统计G中每个度出现的次数
    G_node_num = 0

    GS_degree_distribution = ks.degree_distribution(GS)
    GS_degree_dict = {}
    GS_node_num = 0

    com_degree_set = set([])

    D_KL = 0  # KL散度

    for degree in G_degree_distribution:
        if degree in G_degree_dict:
            G_degree_dict[degree] += 1
        else:
            G_degree_dict[degree] = 1

    for degree in GS_degree_distribution:
        if degree in GS_degree_dict:
            GS_degree_dict[degree] += 1
        else:
            GS_degree_dict[degree] = 1

    a = 0.99
    # 获取所有可能的度
    for k in G_degree_dict:
        com_degree_set.add(k)
    for k in GS_degree_dict:
        com_degree_set.add(k)

    for d in com_degree_set:
        if d in G_degree_dict:
            P1 = G_degree_dict[d] / len(G.nodes)
        else:
            P1 = 0

        if d in GS_degree_dict:
            P2 = GS_degree_dict[d] / len(GS.nodes)
        else:
            P2 = 0

        P = a * P1 + (1 - a) * P2
        Q = a * P2 + (1 - a) * P1
        D_KL += P * math.log2(P / Q)

    return math.fabs(D_KL)


def hop_plot_KL(G_hop_plot_distribution=[], GS_hop_plot_distribution=[]):
    G_hop_plot_dict = {}  # 统计G中每个跳数出现的次数
    G_hop_plot_num = 0  # G中跳数对的总个数

    GS_hop_plot_dict = {}  # 统计GS中每个跳数出现的次数
    GS_hop_plot_num = 0  # GS中跳数对的总个数

    com_hop_plot_set = set([])

    D_KL = 0  # KL散度

    for hop_plot in G_hop_plot_distribution:
        if hop_plot in G_hop_plot_dict:
            G_hop_plot_dict[hop_plot] += 1
        else:
            G_hop_plot_dict[hop_plot] = 1

    for hop_plot in GS_hop_plot_distribution:
        if hop_plot in GS_hop_plot_dict:
            GS_hop_plot_dict[hop_plot] += 1
        else:
            GS_hop_plot_dict[hop_plot] = 1

    a = 0.99
    # 获取所有可能的hop_plot
    for k in G_hop_plot_dict:
        com_hop_plot_set.add(k)
    for k in GS_hop_plot_dict:
        com_hop_plot_set.add(k)

    for d in com_hop_plot_set:
        if d in G_hop_plot_dict:
            P1 = G_hop_plot_dict[d] / len(G_hop_plot_distribution)
        else:
            P1 = 0

        if d in GS_hop_plot_dict:
            P2 = GS_hop_plot_dict[d] / len(GS_hop_plot_distribution)
        else:
            P2 = 0

        P = a * P1 + (1 - a) * P2
        Q = a * P2 + (1 - a) * P1
        D_KL += P * math.log2(P / Q)

    return math.fabs(D_KL)


def clustering_coefficient_KL(G_clustering_coefficient_distribution=[], GS_clustering_coefficient_distribution=[]):
    G_clustering_coefficient_dict = {}  # 统计G中每个聚类系数出现的次数
    G_clustering_coefficient_num = 0  # G中聚类系数的总个数

    GS_clustering_coefficient_dict = {}  # 统计GS中每个聚类系数出现的次数
    GS_clustering_coefficient_num = 0  # GS中聚类系数的总个数

    com_clustering_coefficient_set = set([])

    D_KL = 0  # KL散度

    for clustering_coefficient in G_clustering_coefficient_distribution:
        if clustering_coefficient in G_clustering_coefficient_dict:
            G_clustering_coefficient_dict[clustering_coefficient] += 1
        else:
            G_clustering_coefficient_dict[clustering_coefficient] = 1

    for clustering_coefficient in GS_clustering_coefficient_distribution:
        if clustering_coefficient in GS_clustering_coefficient_dict:
            GS_clustering_coefficient_dict[clustering_coefficient] += 1
        else:
            GS_clustering_coefficient_dict[clustering_coefficient] = 1

    a = 0.99
    # 获取所有可能的clustering_coefficient
    for k in G_clustering_coefficient_dict:
        com_clustering_coefficient_set.add(k)
    for k in GS_clustering_coefficient_dict:
        com_clustering_coefficient_set.add(k)

    for d in com_clustering_coefficient_set:
        if d in G_clustering_coefficient_dict:
            P1 = G_clustering_coefficient_dict[d] / len(G_clustering_coefficient_distribution)
        else:
            P1 = 0

        if d in GS_clustering_coefficient_dict:
            P2 = GS_clustering_coefficient_dict[d] / len(GS_clustering_coefficient_distribution)
        else:
            P2 = 0

        P = a * P1 + (1 - a) * P2
        Q = a * P2 + (1 - a) * P1
        D_KL += P * math.log2(P / Q)

    return math.fabs(D_KL)


def k_core_KL(G_k_core_distribution=[], GS_k_core_distribution=[]):
    G_k_core_dict = {}  # 统计G中每个kcore出现的次数
    G_k_core_num = 0  # G中kcore的总个数

    GS_k_core_dict = {}  # 统计GS中每个kcore出现的次数
    GS_k_core_num = 0  # GS中kcore的总个数

    com_k_core_set = set([])

    D_KL = 0  # KL散度

    for k_core in G_k_core_distribution:
        if k_core in G_k_core_dict:
            G_k_core_dict[k_core] += 1
        else:
            G_k_core_dict[k_core] = 1

    for k_core in GS_k_core_distribution:
        if k_core in GS_k_core_dict:
            GS_k_core_dict[k_core] += 1
        else:
            GS_k_core_dict[k_core] = 1

    a = 0.99
    # 获取所有可能的k_core
    for k in G_k_core_dict:
        com_k_core_set.add(k)
    for k in GS_k_core_dict:
        com_k_core_set.add(k)

    for d in com_k_core_set:
        if d in G_k_core_dict:
            P1 = G_k_core_dict[d] / len(G_k_core_distribution)
        else:
            P1 = 0

        if d in GS_k_core_dict:
            P2 = GS_k_core_dict[d] / len(GS_k_core_distribution)
        else:
            P2 = 0

        P = a * P1 + (1 - a) * P2
        Q = a * P2 + (1 - a) * P1
        D_KL += P * math.log2(P / Q)

    return math.fabs(D_KL)
