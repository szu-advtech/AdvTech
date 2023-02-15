import networkx as nx
import numpy as np
import Util as util
import RW
import math
import SamplingMethods as sm
import SGP
import KStest as ks


# 获得连通分量
def getSubGraphs(G=nx.Graph()):
    subGraphs = {}
    subGraphsNum = 0

    connected_components = list(nx.connected_components(G))
    for component in connected_components:
        subGraphsNum += 1
        subGraph = nx.Graph()
        subGraphs[f"subGraph({subGraphsNum})"] = subGraph
        for i in component:
            subGraph.add_node(i)

    for k in subGraphs:
        for u in subGraphs[k].nodes:
            for v in subGraphs[k].nodes:
                if G.has_edge(u, v):
                    subGraphs[k].add_edge(u, v)

    return subGraphs


def sortG(G=nx.Graph()):
    sortG = nx.Graph()
    node_list = list(G.nodes)
    node_list.sort()

    for node in node_list:
        sortG.add_node(node)

    for v in sortG.nodes:
        for u in sortG.nodes:
            if G.has_edge(v, u):
                sortG.add_edge(v, u)

    return sortG


def getRJStationaryProbability(G=nx.Graph()):
    n = np.zeros((len(G.nodes), len(G.nodes)))
    G = util.sortG(G)
    d = 0.85

    for v in G.nodes:
        v_neighbor = set(G.neighbors(v))
        for u in G.nodes:
            if u in v_neighbor:
                n[v - 1][u - 1] = (1 - d) / len(G.nodes) + d / G.degree[v]
            else:
                n[v - 1][u - 1] = (1 - d) / len(G.nodes)

    a = np.matmul(n, n)
    for i in range(1000):
        a = np.matmul(a, n)
    print(a)


def getRWStationaryProbability(G=nx.Graph()):
    n = np.zeros((len(G.nodes), len(G.nodes)))
    G = util.sortG(G)

    for v in G.nodes:
        v_neighbor = set(G.neighbors(v))
        for u in G.nodes:
            if u in v_neighbor:
                n[v - 1][u - 1] = 1 / G.degree[v]

    a = np.matmul(n, n)
    for i in range(100):
        a = np.matmul(a, n)
    print(a)


def RW_getAvgDegree(G=nx.Graph()):
    GS = RW.RW(G, 0.2)
    b = 0
    for n in GS.nodes:
        b += 1 / G.degree[n]
    return 1 / b * len(GS.nodes)


def MHRW_getAvgDegree(G=nx.Graph()):
    GS = RW.Meropolis_Hastings_RW(G.copy())
    b = 0
    for n in GS.nodes:
        b += G.degree[n]
    print(b / len(GS.nodes))
    return b / len(GS.nodes)


def HHestimator_RW(G=nx.Graph()):
    a = 0
    for n in G.nodes:
        a += G.degree[n]
    tureValue = a / len(G.nodes)
    print(tureValue)

    for j in range(0, 20):
        sum = 0
        for i in range(0, 20):
            sum += RW_getAvgDegree(G)
        print(f"{sum / 20}---{math.fabs(sum / 20 - tureValue) / tureValue}")


def HHestimator_MHRW(G=nx.Graph()):
    a = 0
    for n in G.nodes:
        a += G.degree[n]
    tureValue = a / len(G.nodes)
    print(tureValue)

    for j in range(0, 20):
        sum = 0
        for i in range(0, 20):
            sum += MHRW_getAvgDegree(G)
        print(f"{sum / 20}---{math.fabs(sum / 20 - tureValue) / tureValue}")


def get_GS_diameter(G=nx.Graph()):
    GS1 = RW.Meropolis_Hastings_RW(G.copy())
    GS2 = RW.Meropolis_Hastings_RJ(G.copy())
    GS3 = RW.RW(G.copy())
    GS4 = RW.RJ(G.copy())
    GS5 = sm.FFS(G.copy())
    GS6 = SGP.SGP(G.copy())
    GS7 = sm.VS(G.copy())
    GS8 = sm.ES(G.copy())

    print(nx.diameter(GS1))

    node_nums = 0
    subGraphs = getSubGraphs(GS2)
    for k in subGraphs:
        if len(subGraphs[k]) > node_nums:
            GS2 = subGraphs[k]
            node_nums = len(subGraphs[k])
    print(nx.diameter(GS2))

    print(nx.diameter(GS3))

    node_nums = 0
    subGraphs = getSubGraphs(GS4)
    for k in subGraphs:
        if len(subGraphs[k]) > node_nums:
            GS4 = subGraphs[k]
            node_nums = len(subGraphs[k])
    print(nx.diameter(GS4))

    node_nums = 0
    subGraphs = getSubGraphs(GS5)
    for k in subGraphs:
        if len(subGraphs[k]) > node_nums:
            GS5 = subGraphs[k]
            node_nums = len(subGraphs[k])
    print(nx.diameter(GS5))

    node_nums = 0
    subGraphs = getSubGraphs(GS6)
    for k in subGraphs:
        if len(subGraphs[k]) > node_nums:
            GS6 = subGraphs[k]
            node_nums = len(subGraphs[k])
    print(nx.diameter(GS6))

    node_nums = 0
    subGraphs = getSubGraphs(GS7)
    for k in subGraphs:
        if len(subGraphs[k]) > node_nums:
            GS7 = subGraphs[k]
            node_nums = len(subGraphs[k])
    print(nx.diameter(GS7))

    node_nums = 0
    subGraphs = getSubGraphs(GS8)
    for k in subGraphs:
        if len(subGraphs[k]) > node_nums:
            GS8 = subGraphs[k]
            node_nums = len(subGraphs[k])
    print(nx.diameter(GS8))
