import random
import numpy as np
import Graphstructure as G
import queue as q

'''  
深度优先搜索实现节点采样：开始的节点是随机选择的，邻居是通过不同顺序加入到后进先出队列（栈）中的。
'''

def set_seed(seed):
    """随机种子设置"""
    random.seed(seed)
    np.random.seed(seed)

def check_sample_size(newgraph,graph):
    """判断抽样图是否小于原图"""
    if G.get_number_of_nodes(newgraph) > G.get_number_of_nodes(graph):
        raise ValueError("The number of nodes is too large. Please see requirements.")
    if G.get_number_of_edges(newgraph) > G.get_number_of_edges(graph):
        raise ValueError("The number of edges is too large. Please see requirements.")


def create_queue(graph, start_node):
    """
    （初始化）创建一个受访节点集和一个遍历路径列表。
    输入原图以及初始节点，也可以不输入初始节点，让系统随机选择
    """
    queue = q.LifoQueue()# 创建受访节点集，后进先出队列
    if start_node is not None:# 开始点==非空，则随机抽取开始节点
    # 开始节点必定是在1~节点数之间
        if start_node >= 0 and start_node < G.get_number_of_nodes(graph):
            queue.put(start_node)
        else:
            raise ValueError("Starting node index is out of range.")
    else:# 开始点==空，则随机抽取开始点
        start_node = random.choice(range(G.get_number_of_nodes(graph)))
        queue.put(start_node)
        return queue

def extract_edges(path):
    # 从深度搜索树中提取边,实际上，就是按照遍历路径创建边
    edges = [[path[i], path[i + 1]] for i in range(len(path) - 1)]
    return edges

'''********************该函数把ModDFS抽取节点以及inducetion获得子图*****************************'''
def ModDFS_induce(graph, start_node=None,rate=0.01,seed=42):
    '''为了有重复性，可以设置随机种子'''
    set_seed(seed)
    '''检查输入类型是否符合要求的类型'''
    # if G.check_graph(graph) != 'OK':
    #     raise ValueError("The graph is not suitable.")
    ''' 开始抽样'''
    number_of_samples = round(G.get_number_of_nodes(graph) * rate)  # 计算抽样节点数目
    # 初始化节点序列
    queue=create_queue(graph, start_node)
    nodes = set()  # 创建抽取节点集（无序）
    path = []  # 创建遍历路径序列（有序），虽然也是点集，但是为了连接边而创建
    # 开始抽样
    while len(nodes) < number_of_samples:
        source = queue.get()  # 从队列里取节点
        if source not in nodes:  # source是下一步的起始节点，只有节点没有被取过才能作为source
            # source如果已经被遍历过，if 不成立就会跳回上一步重新在队列里面去节点
            neighbors = G.rank_neighbor(graph, source)  #获得已经按照度顺序排列好邻节点集
            for neighbor in neighbors:  # 将现节点的所有邻居加进队列
                queue.put(neighbor)
            nodes.add(source)
    newgraph = G.get_subgraph(graph, nodes)  # 这一步是为了除去因为跳跃所带来的多余边
    return newgraph

'''**************************该函数ModDFS获得子图*************************************'''
def ModDFS(graph, start_node=None,rate=0.01,seed=42):
    '''为了有重复性，可以设置随机种子'''
    set_seed(seed)
    '''检查输入类型是否符合要求的类型'''
    if G.check_graph(graph) != 'OK':
        raise ValueError("The graph is not suitable.")
    ''' 开始抽样'''
    number_of_samples = round(G.get_number_of_nodes(graph) * rate)  # 计算抽样节点数目
    # 初始化节点序列
    queue=create_queue(graph, start_node)
    nodes = set()  # 创建抽取节点集（无序）
    path = []  # 创建遍历路径序列（有序），虽然也是点集，但是为了连接边而创建
    # 开始抽样
    while len(nodes) < number_of_samples:
        source = queue.get()  # 从队列里取节点
        if source not in nodes:  # source是下一步的起始节点，只有节点没有被取过才能作为source
            # source如果已经被遍历过，if 不成立就会跳回上一步重新在队列里面去节点
            neighbors = G.rank_neighbor(graph, source)  #获得已经按照度顺序排列好邻节点集
            for neighbor in neighbors:  # 将现节点的所有邻居加进队列
                queue.put(neighbor)
            nodes.add(source)
            path.append(source)
    edges=extract_edges(path)  # 节点已经加好了，可以按照路径加边
    if len(edges) > 0:
        newgraph = G.graph_from_edgelist(edges)
        newgraph = G.get_subgraph(newgraph, nodes)  # 这一步是为了除去因为跳跃所带来的多余边
    else:
        newgraph = G.get_subgraph(graph, nodes)
    return newgraph

'''**************************该函数DFS获得子图*************************************'''
def DFS(graph, start_node=None,rate=0.01,seed=42):
    '''为了有重复性，可以设置随机种子'''
    set_seed(seed)
    '''检查输入类型是否符合要求的类型'''
    if G.check_graph(graph) != 'OK':
        raise ValueError("The graph is not suitable.")
    ''' 开始抽样'''
    number_of_samples = round(G.get_number_of_nodes(graph) * rate)  # 计算抽样节点数目
    # 初始化节点序列
    queue=create_queue(graph, start_node)
    nodes = set()  # 创建抽取节点集（无序）
    path = []  # 创建遍历路径序列（有序），虽然也是点集，但是为了连接边而创建
    # 开始抽样
    while len(nodes) < number_of_samples:
        source = queue.get()  # 从队列里取节点
        if source not in nodes:  # source是下一步的起始节点，只有节点没有被取过才能作为source
            # source如果已经被遍历过，if 不成立就会跳回上一步重新在队列里面去节点
            neighbors = G.get_neighbors(graph, source)
            random.shuffle(neighbors)  # 洗牌是为了可以随机从邻接点里面选择下一个节点
            for neighbor in neighbors:  # 将现节点的所有邻居加进队列
                queue.put(neighbor)
            nodes.add(source)
            path.append(source)
    edges=extract_edges(path)  # 节点已经加好了，可以按照路径加边
    if len(edges) > 0:
        newgraph = G.graph_from_edgelist(edges)
        newgraph = G.get_subgraph(newgraph, nodes)  # 这一步是为了除去因为跳跃所带来的多余边
    else:
        newgraph = G.get_subgraph(graph, nodes)
    return newgraph