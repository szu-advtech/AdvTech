from collections import defaultdict
import networkx as nx
import pandas as pd

def read_graph(graphname):
    datadir = "gen_graph/"

    graphpath = str(datadir) + str(graphname) + ".txt"
    f = open(graphpath, 'r')
    line = f.readline()
    G = Graph()
    edges_set = set()

    while line:
        data = line[:-1].split(' ')
        u = int(data[0])
        v = int(data[1])
        if u == v or (u, v) in edges_set or (v, u) in edges_set:
            line = f.readline()
            continue
        G.nlist[u].append(v)
        G.nlist[v].append(u)
        edges_set.add((u, v))
        line = f.readline()

    f.close()

    G.nodes = list(G.nlist.keys())
    G.N = len(G.nodes)

    m = 0  # 边数
    for v in G.nodes:
        d = int(len(G.nlist[v]))
        m += d
        if d > G.maxd:
            # 最大度
            G.maxd = d

    G.M = int(m / 2)

    print('Read ' + str(graphname) + " graph.")
    print("Number of nodes: " + str(G.N))
    print("Number of edges: " + str(G.M))

    return G

def convert_to_Graph_of_networkx(G):
    # 用 networkx 保存图
    nxG = nx.Graph()

    for v in G.nodes:
        for w in G.nlist[v]:
            nxG.add_edge(v, w)

    return nxG


class Graph():

    def __init__(self):
        self.nodes = set()  # set of nodes 节点集合
        self.qry_nodes = set()  # set of queried nodes 查询节点集合
        self.vis_nodes = set()  # set of visible nodes 邻居可访问节点集合
        self.nlist = defaultdict(list)  # lists of neighbors 节点的邻居节点列表

        # graph properties
        self.N = 0  # number of nodes 节点数量
        self.M = 0  # number of edges 边数量
        self.maxd = 0  # maximum degree 最大度
        self.aved = 0  # average degree 平均度
        self.acc = 0  # average clustering coefficient 平均聚类系数
        self.apl = 0  # average shortest path length 平均最短路径长度
        self.diameter = 0  # diameter 直径
        self.lambda_1 = 0  # 最大特征值
        self.dd = defaultdict(float)  # degree distribution 度分布
        self.num_deg = defaultdict(int)  # number of nodes with degrees 度为k的节点数量
        self.jdd = defaultdict(lambda: defaultdict(float))  # joint degree distribution 联合度分布，二维
        self.knn = defaultdict(float)  # neighbor connectivity 邻域连通性
        self.num_tri = defaultdict(int)  # number of triangles of nodes 节点的三角形数量
        self.ddcc = defaultdict(float)  # degree-dependent clustering coefficient 度依赖聚类系数
        self.cnd = defaultdict(float)  # common neighbor distribution 公共邻居分布：两个节点都连接到的第三个节点（公共节点）
        self.spld = defaultdict(float)  # shortest path length distribution 最短路径长度分布
        self.ddbc = defaultdict(float)  # degree-dependent betweeness centrality 度依赖的中介中心性


if __name__ == '__main__':
    # graphname = "Slashdot0811"
    # G = read_graph(graphname)
    # nxG = convert_to_Graph_of_networkx(G)
    # d = nx.degree(nxG)
    # a = sorted(d, key=lambda x: x[1])
    # largest = max(nx.connected_components(nxG), key=len)
    # largest_component = nxG.subgraph(largest)
    # print()

    df = pd.read_csv("gen_graph/Slashdot0811.txt", delimiter=" ")
    df.to_csv("picture_data/gen-Slashdot0811.csv", encoding='utf-8', index=False)
