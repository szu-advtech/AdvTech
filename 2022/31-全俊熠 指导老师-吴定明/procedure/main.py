import graph
import sampling
import generation
import time
import os
import sys


# 获取原始图
def read_graph(path, graph_name, seq):
    # 获取图的路径
    graph_path = str(path) + str(graph_name) + ".txt"
    # 获取图文件
    graph_file = open(graph_path, 'r')
    line = graph_file.readline()
    # 初始化保存图的数据结构
    G = graph.Graph()
    # 防止重复保存边
    edges_set = set()

    while line:
        # 获取边
        edge = line[:-1].split(seq)
        # 边的一端节点
        u = int(edge[0])
        # 边的另一端节点
        v = int(edge[1])
        # 不保存自环和已经保存了的边
        if u == v or (u, v) in edges_set or (v, u) in edges_set:
            line = graph_file.readline()
            continue
        # 保存边
        G.nlist[u].append(v)
        G.nlist[v].append(u)
        # 添加边
        edges_set.add((u, v))
        # 读取下一行数据
        line = graph_file.readline()
    # 关闭io流
    graph_file.close()

    # 图的节点集
    G.nodes = list(G.nlist.keys())
    # 图的节点数量
    G.N = len(G.nodes)

    # 边数
    edge_num = 0
    for v in G.nodes:
        # 节点v的度
        d = int(len(G.nlist[v]))
        edge_num += d
        if d > G.maxd:
            # 最大度
            G.maxd = d
    # 保存边数
    G.M = int(edge_num / 2)

    # 打印读取图的数据
    print('Read ' + str(graph_name) + " graph.")
    print("Number of nodes: " + str(G.N))
    print("Number of edges: " + str(G.M))

    return G


# 保存生成图
def store_genG(graph_name, genG):
    # 创建gen_graph目录
    if not os.path.exists("../gen_graph/"):
        os.mkdir("../gen_graph/")
    # 保存文件路径
    genG_path = "../gen_graph/" + str(graph_name) + ".txt"
    genG_file = open(genG_path, 'w')
    for u in genG.nodes:
        for v in genG.nlist[u]:
            if u <= v:
                genG_file.write(str(u) + " " + str(v) + "\n")
    # 关闭IO
    genG_file.close()

if __name__ == '__main__':
    # graph_name = "Slashdot0811"
    graph_name = "syn10000"
    # 读取原图
    # G = read_graph("../data/", graph_name, '\t')
    G = read_graph("../data/", graph_name, ' ')
    # 采样大小为原图节点数量的10%
    sample_size = int(0.1 * G.N)

    # 从节点集中随机选择一个节点作为随机游走的起始节点
    # seed = sampling.select_seed(G) 392：129705  1915：128425  58：128927  258：123828  277：125745  837:12388
    seed = 258
    # 随机游走得到的子图414
    samplinglist = sampling.random_walk(G, sample_size, seed)

    # 程序运行开始时间
    start = time.time()
    # 通过随机游走得到的子图通过四个步骤重新生成与原图更加接近的生成图
    genG = generation.graph_restoration_method(samplinglist)
    # 程序运行结束时间
    end = time.time()
    # 生成图所需时间
    print("Generation time [sec]: " + str(end - start))

    # 保存生成图
    store_genG(graph_name, genG)

    # 读取生成图
    gen_G = G = read_graph("../gen_graph/", graph_name, ' ')
    # 计算与原图的误差
    graph.calc_error_of_each_proeprty(G, gen_G)



