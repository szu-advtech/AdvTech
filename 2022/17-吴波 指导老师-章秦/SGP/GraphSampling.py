import networkx as nx
import GraphPartition as gp
import random
import math


def getGraphDiameter(G=nx.Graph()):
    diameter = -1
    diameterVertex = {}
    n = 0
    for v in G.nodes:
        for u in G.nodes:
            n += 1
            print(n)
            if nx.shortest_path_length(G, v, u) > diameter:
                diameter = nx.shortest_path_length(G, v, u)
                diameterVertex['v'] = v
                diameterVertex['u'] = u
    return diameterVertex


def stratifiedSampling(G=nx.Graph(), initialG=nx.Graph(), P=0.2):
    GS = nx.Graph()  # 采样图
    sample_k = 0.7

    # diameterVertex = getGraphDiameter(G)
    vs = nx.periphery(G)[0]  # 起始点
    dict = {}  # 存储V,S集合
    diameter = nx.diameter(G)  # 图的直径

    stratified_nodes = nx.single_source_shortest_path_length(G, vs)

    for k in stratified_nodes:
        if f"V{stratified_nodes[k]}" in dict:
            dict[f"V{stratified_nodes[k]}"].append(k)
        else:
            dict[f"V{stratified_nodes[k]}"] = []
            dict[f"V{stratified_nodes[k]}"].append(k)

    for i in list(range(0, diameter + 1)):
        dict[f"S{i}"] = []

    # 将起始点放入S0采样集
    dict['S0'].append(vs)

    # 分层抽样
    for i in list(range(1, diameter + 1)):
        V_io = []
        V_ino = []

        # 拆分Vi集合
        for v in dict[f"V{i}"]:
            flag = 0
            for s in dict[f"S{i - 1}"]:
                if G.has_edge(v, s):
                    V_io.append(v)
                    flag = 1
                    break;

            if flag == 0:
                V_ino.append(v)

        # 总共要抽的顶点
        sample_size = math.floor(len(dict[f"V{i}"]) * P)
        V_io_num = 0
        V_ino_num = 0

        if sample_size == 0:
            if len(V_io) != 0:
                V_io_num = 1
            else:
                V_ino_num = 1

        if sample_size != 0:
            if math.floor(sample_size * sample_k) == 0:
                V_io_num = 1
            else:
                V_io_num = int(sample_size * sample_k)
            if math.floor(sample_size * (1 - sample_k)) == 0:
                V_ino_num = 1
            else:
                V_ino_num = int(sample_size * (1 - sample_k))

        if V_io_num > len(V_io):
            V_io_num = len(V_io)

        if V_ino_num > len(V_ino):
            V_ino_num = len(V_ino)

        dict[f"S{i}"] = random.sample(V_io, V_io_num) + random.sample(V_ino, V_ino_num)

    # 合并抽样集S，获得采样图顶点
    for i in list(range(0, nx.diameter(G) + 1)):
        for v in dict[f"S{i}"]:
            GS.add_node(v)

    # 给采样图增加边
    for u in GS.nodes:
        for v in GS.nodes:
            if initialG.has_edge(u, v):
                GS.add_edge(u, v)

    return GS
