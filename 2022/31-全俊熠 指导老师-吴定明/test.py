from collections import defaultdict
import igraph
# import numpy as np

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


def read_graph(graphname):
    f = open(graphname, 'r')
    line = f.readline()
    G = Graph()
    space = ' '

    while line:
        data = line[:-1].split(space)
        u = int(data[0])
        v = int(data[1])
        G.nlist[u].append(v)
        G.nlist[v].append(u)
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

def calc_knn(G: Graph):
    # 计算邻域连通性
    V_k = defaultdict(list)  # {度k：[度为k的节点集]}

    for v in list(G.nodes):
        k = len(G.nlist[v])
        V_k[k].append(v)

    G.knn = defaultdict(float)
    for k in V_k:
        if k * len(V_k[k]) == 0:
            continue
        for v in V_k[k]:
            for w in G.nlist[v]:
                G.knn[k] += len(G.nlist[w])  # 度为k的节点的邻居节点的度的总和
        G.knn[k] = float(G.knn[k]) / (k * len(V_k[k]))  # 总和 / (k * 度为k的节点数)

    return


def calc_clustering(G: Graph):
    # 计算图的聚类系数

    V_d = defaultdict(int)  # 存放度k的数量：{度k，n_k}
    sum_lcc_d = defaultdict(float)  # 存放度为d的节点的聚类系数总和
    sum_lcc = 0  # 存放图中所有节点的聚类系数总和

    V = list(G.nodes)

    for v in V:
        d = int(len(G.nlist[v]))
        V_d[d] += 1

        if d == 0 or d == 1:  # 孤立节点不用参加聚类系数的计算
            continue

        lcc = 0  # 局部聚类系数
        for i in range(0, d - 1):
            x = G.nlist[v][i]  # 节点v的邻居节点x
            for j in range(i + 1, d):
                y = G.nlist[v][j]  # 节点v的另一个邻居节点y
                if v != x and x != y and y != v:  # 三个不同节点
                    lcc += 2 * G.nlist[x].count(y)  # 节点x与节点y存在边，局部聚类系数+2

        lcc = float(lcc) / (d * (d - 1))  # 度为d的聚类系数
        sum_lcc_d[d] += lcc  # 度为d的总聚类系数
        sum_lcc += lcc  # 总聚类系数

    G.ddcc = defaultdict(float)
    for d in V_d:
        if V_d[d] > 0:
            G.ddcc[d] = float(sum_lcc_d[d]) / V_d[d]  # 度依赖聚类系数

    N = len(V)
    G.acc = float(sum_lcc) / N  # 平均聚类系数

    return


def calc_common_neighbor_distribution(G: Graph):
    # 计算公共邻居分布
    G.cnd = defaultdict(float)

    for i in G.nodes:  # 节点i作为公共节点
        for j in G.nlist[i]:  # 遍历节点i的邻居节点
            if j <= i:  # 比节点i下标小的都访问过了
                continue
            m = 0  # 以节点i为公共节点的数量
            for k in G.nlist[i]:
                if k == i and k == j:  # 第三个节点要与前面两个节点不同
                    continue
                m += G.nlist[j].count(k)
            G.cnd[m] += 1  # 公共节点数量为m的顶点数+1

    norm = sum(list(G.cnd.values()))  # 求总数用于归一化
    for m in G.cnd:
        G.cnd[m] = float(G.cnd[m]) / norm

    return


def convert_to_Graph_of_igraph(G: Graph):
    # 用 igraph 保存图
    iG = igraph.Graph()
    edges = []

    for v in G.nodes:
        for w in G.nlist[v]:
            if w >= v:
                edges.append([v, w])

    iG.add_vertices(len(G.nodes))
    iG.add_edges(edges)

    return iG


def calc_shortest_path_properties(G: Graph):
    # Note: calculate shortest path properties of the largest connected component of a given graph.
    # 计算最短路径

    iG = convert_to_Graph_of_igraph(G)  # 用将图G转换为igraph图
    igraph_path_length_hist = iG.path_length_hist(directed=False)  # 返回Histogram对象，用于存储未连接的顶点对的数量

    G.spld = defaultdict(float)  # key是路径长度，value是数量
    num_all = igraph_path_length_hist.n  # 为连接的顶点对数量
    bins = tuple(igraph_path_length_hist.bins())  # (左边界，右边界，仓库内元素的数量)

    for (i, j, k) in bins:
        if j != i + 1:  # 如果右边界与左边界差距不是1，就报错
            print("Error.")
            exit(0)
        G.spld[i] = float(k) / num_all  #

    G.diameter = max(list(dict(G.spld).keys()))  # 图的直径：最大路径长度
    G.apl = sum([l * G.spld[l] for l in G.spld])  # 图的平均路径长度

    return


def calc_betweenness(G: Graph):
    # Note: calculate shortest path of the largest connected component of a given graph.
    # 计算度依赖的中介中心性

    iG = convert_to_Graph_of_igraph(G)
    degrees = iG.degree(list(range(0, len(G.nodes))))  # 返回每个节点的度
    bc = iG.betweenness(directed=False)  # 计算中介中心性
    n = int(iG.vcount())  # 节点数量

    G.ddbc = defaultdict(float)
    V_d = defaultdict(int)
    for i in range(0, len(degrees)):
        d = degrees[i]  # 节点i的度
        G.ddbc[d] += float(bc[i]) / ((n - 1) * (n - 2))  # 度为d的中介中心性
        V_d[d] += 1  # 度为d的数量加1

    for d in G.ddbc:
        G.ddbc[d] = float(G.ddbc[d]) / V_d[d]  # 求平均

    return

def calc_largest_eigenvalue(G: Graph):
    # 计算图的最大特征值
    iG = convert_to_Graph_of_igraph(G)
    L = iG.laplacian(normalized=True)  # 返回图的 Laplacian matrix
    eigenvalues = np.linalg.eigvals(L)  # 计算矩阵的置
    G.lambda_1 = float(max(eigenvalues))  # 最大特征值

    return


if __name__ == '__main__':
    # file = "data/syn10000.txt"
    # g = read_graph(file)
    # calc_largest_eigenvalue(g)
    # print(g.acc)

    print(11)