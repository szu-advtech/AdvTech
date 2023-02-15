import random
import numpy as np
import networkx as nx
import pandas as pd

def getgraph(file):
    '''初始化图结构'''
    edgelist = pd.read_csv(file)
    graph = nx.convert_matrix.from_pandas_edgelist(edgelist, "id_1", "id_2")
    return graph

def check_directedness(graph):
    '''判断是否为有向图'''
    if nx.is_directed(graph) == False:
        return 0

def check_indexing(graph):
    '''判断节点编号是否连续'''
    numeric_indices = [index for index in range(graph.number_of_nodes())]
    node_indices = sorted([node for node in graph.nodes()])
    if numeric_indices == node_indices:
        return 0

def check_graph(graph):
    '''输入必须节点编号合法且无向图'''
    if check_directedness(graph)==check_indexing(graph)==0:
        return 'OK'

def get_number_of_nodes(graph):
    '''计算点数'''
    return nx.number_of_nodes(graph)

def get_number_of_edges(graph):
    '''计算边数'''
    return nx.number_of_edges(graph)

def get_nodes(graph):
    '''获得全图节点'''
    return [node for node in nx.nodes(graph)]

def get_edges(graph,bunch=None):
    '''获得全图边集或者与v有关的边集'''
    return [edge for edge in nx.edges(graph,bunch)]

def graph_from_edgelist(edges):
    '''给定边集创建图'''
    graph = nx.from_edgelist(edges)
    return graph

def get_subgraph(graph,nodes):
    '''给定节点返回子图'''
    return nx.subgraph(graph,nodes)

def deleteedge(graph,edge):
    '''从图中删边'''
    graph.remove_edge(edge[0],edge[1])
    return graph

def get_neighbors(graph,node):
    '''获得节点的邻接点'''
    return [node for node in nx.neighbors(graph,node)]

def get_random_neighbor(graph,node):
    '''返回节点的任意一个邻接点'''
    neighbors = get_neighbors(graph, node)
    return random.choice(neighbors)

def rank_neighbor(graph, source):
    '''按将邻接点按度顺序排列'''
    neighbors = get_neighbors(graph, source)
    neighbors_num = len(neighbors)
    neighbors_degree = []
    for i in range(0, neighbors_num - 1):
        neighbors_degree.append(get_degree(graph, neighbors[i]))
    z = list(zip(neighbors, neighbors_degree))
    z = sorted(z, key=(lambda x: x[1]))
    i = 0  # 按第一列排序
    k = [item[i] for item in z]  # 结果：[1, 3]
    return k

def get_degree(graph,node=None):
    '''获得指定节点的度'''
    return nx.degree(graph,node)

def get_clusteringcoefficient(graph,node=None):
    '''获得给定图的平均聚类系数'''
    return nx.clustering(graph,node)

def AVGdegree(graph):
    '''获得全图平均度'''
    degree=dict(nx.degree(graph))
    return sum(degree.values())/get_number_of_nodes(graph)

def AVGclusteringcoefficient(graph):
    '''获得给定图的平均聚类系数'''
    return nx.average_clustering(graph)

def AVGshortestpathlength(graph):
    '''平均路径长度'''
    return nx.average_shortest_path_length(graph)

def diameter(graph):
    '''图直径'''
    return nx.diameter(graph)

def assortativity(graph):
    '''图同配性'''
    return nx.degree_assortativity_coefficient(graph)

def cc_distribution(graph):
    cluster = nx.clustering(graph)
    cluster = dict(cluster)
    cluster=list(cluster.values())
    # cc=pd.Series(cluster)
    # cc=dict(cc.value_counts(normalize=True))
    return cluster

def degree_distribution(graph):
    degree = nx.degree(graph)
    degree = dict(degree)
    degree=degree.values()
    degree=pd.Series(degree)
    degree=dict(degree.value_counts(normalize=True))
    return degree

def shortestpath_distribution(graph):
    sp = nx.shortest_path_length(graph)
    sp = dict(sp)
    sp=sp.values()
    sp=pd.Series(sp)
    sp=dict(sp.value_counts(normalize=True))
    return sp
