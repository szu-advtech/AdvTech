import networkx as nx
import random
import math
from scipy import stats
import queue


# 森林火灾抽样
def FFS(G=nx.Graph(), P=0.2):
    GS = nx.Graph()
    num = 0  # 抽取的结点数量
    total_num = len(list(G.nodes))

    for v in G:
        G.nodes[v]['isFired'] = 0

    V = []
    while 1:
        if num > total_num * 0.2:
            break
        # print(num)
        if len(V) == 0:
            start = random.sample(list(G.nodes), 1)[0]
            # while G.nodes[start]['isFired'] != 0:
            #     start = random.sample(list(G.nodes), 1)[0]
            V.append(start)
            G.nodes[start]['isFired'] = 1
            GS.add_node(start)
            num += 1

        V_next = []
        for u in V:
            if num > total_num * P:
                break

            N = list(G.neighbors(u))
            if len(N) == 0:
                continue

            k = stats.geom.rvs(p=0.3) - 1  # 当前结点燃烧的边数量
            while k < 1 or k > len(N):
                k = stats.geom.rvs(p=0.3) - 1
                # print(f"k={k},len={len(N)}")

            N_fire = random.sample(N, k)  # 燃烧的邻居结点
            for v in N_fire:
                GS.add_edge(u, v)
                if G.nodes[v]['isFired'] == 0:
                    V_next.append(v)
                    G.nodes[v]['isFired'] = 1
                    num += 1

        # print(f"V={V}")
        # print(f"V_next={V_next}")
        for u in V:
            G.remove_node(u)
        V = V_next
    return GS


# 顶点采样
def VS(G=nx.Graph(), P=0.2):
    GS = nx.Graph()
    GS_node_list = random.sample(list(G.nodes), int(len(list(G.nodes)) * P))  # 随机抽取P个百分点的顶点

    for v in GS_node_list:
        GS.add_node(v)

    for v in GS_node_list:
        for u in GS_node_list:
            if G.has_edge(u, v):
                GS.add_edge(u, v)

    return GS


# 边采样
def ES(G=nx.Graph(), P=0.2):
    GS = nx.Graph()
    edges = random.sample(list(G.edges), int(len(list(G.nodes)) * P / 2))
    for e in edges:
        GS.add_edge(e[0], e[1])

    while len(list(GS.nodes)) < len(list(G.nodes)) * P:
        edge = random.sample(list(G.edges), 1)
        GS.add_edge(edge[0][0], edge[0][1])

    return GS
