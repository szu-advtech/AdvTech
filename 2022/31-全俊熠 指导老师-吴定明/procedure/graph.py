import networkx as nx
import igraph
from collections import Counter
from collections import defaultdict
from collections import deque
import math
import numpy as np
import warnings

warnings.simplefilter("ignore", np.ComplexWarning)


class Graph():

    def __init__(self):
        self.nodes = set()  # set of nodes 节点集合
        self.qry_nodes = set()  # set of queried nodes 查询节点集合
        self.vis_nodes = set()  # set of visible nodes 邻居可访问节点集合
        self.nlist = defaultdict(list)  # lists of neighbors 节点的邻居节点列表

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


def convert_to_Graph_of_networkx(G: Graph):
    # 用 networkx 保存图
    nxG = nx.Graph()

    for v in G.nodes:
        for w in G.nlist[v]:
            nxG.add_edge(v, w)

    return nxG


def convert_to_MultiGraph_of_networkx(G: Graph):
    # 多图
    nxG = nx.MultiGraph()

    for v in G.nodes:
        for w in G.nlist[v]:
            nxG.add_edge(v, w)

    return nxG


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


def is_connected(G: Graph):
    # 图是否连通
    Q = deque()  # 双向队列
    V = list(G.nodes)
    visit = defaultdict(int)

    v = V[0]
    Q.append(v)
    visit[v] = 1
    n = 0

    # bfs
    while len(Q) > 0:
        v = Q.popleft()
        n += 1

        for w in G.nlist[v]:
            if visit[w] == 0:
                visit[w] = 1
                Q.append(w)

    return n == len(V)


def largest_connected_component(G: Graph):
    # 返回最大连通子图
    search = {v: 0 for v in list(G.nodes)}
    LCC_nodes = []  # 最大连通子图的节点集
    LCC_nlist = defaultdict(list)  # 最大连图子图的结构 {节点u：u的邻居节点集,...}
    n = 0  # 最大连通子图的节点数量
    N = len(G.nodes)  # 总数

    for v in G.nodes:  # 从图的每一个节点出发
        if sum(list(search.values())) >= N - n:
            break

        if search[v] == 1:  # 节点v所在的连通图访问过了
            continue

        Q = deque()
        visit = defaultdict(int)  # 用于判断节点是否访问过
        visit[v] = 1
        Q.append(v)
        CC_nodes = []  # 节点v所在的连通图的节点集
        CC_nlist = defaultdict(list)  #

        # bfs
        while len(Q) > 0:
            u = Q.popleft()
            CC_nodes.append(u)
            search[u] = 1  # 节点u所在的连通图已经访问
            for w in G.nlist[u]:  # 访问节点u的邻居节点集
                CC_nlist[u].append(w)
                if visit[w] == 0:
                    visit[w] = 1
                    Q.append(w)

        if len(CC_nodes) > n:  # 当前的连通子图如果大于最大连通子图
            n = len(CC_nodes)
            LCC_nodes = list(CC_nodes)
            LCC_nlist = defaultdict(CC_nlist)

    LCC = Graph()
    LCC.nodes = list(LCC_nodes)
    LCC.N = len(LCC.nodes)
    LCC.nlist = defaultdict(LCC_nlist)

    m = 0
    for v in LCC.nodes:
        d = int(len(LCC.nlist[v]))
        m += d
        if d > LCC.maxd:
            LCC.maxd = d
    LCC.M = int(m) / 2

    return LCC


def add_edge(G: Graph, u, v):
    # 添加边
    G.nlist[u].append(v)
    G.nlist[v].append(u)

    return G


def remove_edge(G: Graph, u, v):
    # 移除边
    G.nlist[u].remove(v)
    G.nlist[v].remove(u)

    return G


def calc_dd(G: Graph):
    # 计算度分布
    V = list(G.nodes)
    n = len(V)
    degrees = {v: int(len(G.nlist[v])) for v in V}  # {节点v：度数}
    num_deg = Counter(list(degrees.values()))  # {度k：度k的数量}
    G.dd = defaultdict(float)
    for d in num_deg:
        G.dd[d] = float(num_deg[d]) / n  # 度k的数量 / 节点总数

    return


def calc_num_deg(G: Graph):
    # {度k：度k的数量}
    V = list(G.nodes)
    degrees = {v: int(len(G.nlist[v])) for v in V}
    G.num_deg = Counter(list(degrees.values()))

    return


def calc_jdd(G: Graph):
    # 计算联合度分布 度为k的所有节点与度为l的所有节点之间的边数 /  (2*图边的总数)
    G.jdd = defaultdict(lambda: defaultdict(float))
    V = list(G.nodes)

    for v in V:
        k = int(len(G.nlist[v]))
        for w in list(G.nlist[v]):
            l = int(len(G.nlist[w]))
            G.jdd[k][l] += 1

    for k in G.jdd:
        for l in G.jdd[k]:
            G.jdd[k][l] = float(G.jdd[k][l]) / (2 * G.M)

    return


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


def calc_num_tri(G: Graph):
    # 计算节点的三角形数量
    G.num_tri = defaultdict(int)
    V = list(G.nodes)

    for v in V:
        d = int(len(G.nlist[v]))
        if d == 0 or d == 1:
            continue

        n_t = 0  # 节点v的三角形数量
        for i in range(0, d - 1):
            x = G.nlist[v][i]
            for j in range(i + 1, d):
                y = G.nlist[v][j]
                if v != x and x != y and y != v:
                    n_t += G.nlist[x].count(y)

        G.num_tri[d] += n_t

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


def calc_properties(G: Graph):
    G.N = len(G.nodes)

    G.M = 0
    for i in G.nodes:
        G.M += len(G.nlist[i])
    G.M = int(G.M) / 2

    G.aved = float(2 * G.M) / G.N

    calc_dd(G)

    calc_knn(G)

    calc_clustering(G)

    calc_common_neighbor_distribution(G)

    calc_shortest_path_properties(G)

    calc_betweenness(G)

    # calc_largest_eigenvalue(G)

    return


def calc_relative_error_for_scalar_property(G_property, genG_property):
    return float(math.fabs(G_property - genG_property)) / G_property


def calc_normalized_L1_distance_for_distribution(G_property, genG_property):
    keys = set(G_property.keys()) | set(genG_property.keys())
    dist = 0
    norm = sum(list(G_property.values()))
    for key in keys:
        dist += math.fabs(G_property[key] - genG_property[key])

    return float(dist) / norm


def calc_error_of_each_proeprty(G: Graph, genG: Graph):
    calc_properties(G)
    calc_properties(genG)

    print("Normalized L1 distance of each property of a generated graph.")

    print("Number of nodes:", calc_relative_error_for_scalar_property(G.N, genG.N))

    print("Average degree:", calc_relative_error_for_scalar_property(G.aved, genG.aved))

    print("Degree distribution:", calc_normalized_L1_distance_for_distribution(G.dd, genG.dd))

    print("Neighbor connectivity:", calc_normalized_L1_distance_for_distribution(G.knn, genG.knn))

    print("Average local clustering coefficient:", calc_relative_error_for_scalar_property(G.acc, genG.acc))

    print("Degree-dependent clustering coefficient:", calc_normalized_L1_distance_for_distribution(G.ddcc, genG.ddcc))

    print("Common neighbor distribution:", calc_normalized_L1_distance_for_distribution(G.cnd, genG.cnd))

    print("Average shortest path length:", calc_relative_error_for_scalar_property(G.apl, genG.apl))

    print("Shortest path length distribution:", calc_normalized_L1_distance_for_distribution(G.spld, genG.spld))

    print("Diameter:", calc_relative_error_for_scalar_property(G.diameter, genG.diameter))

    print("Degree-dependent betweenness centrality:", calc_normalized_L1_distance_for_distribution(G.ddbc, genG.ddbc))

    # print("Largest eigenvalue:", calc_relative_error_for_scalar_property(G.lambda_1, genG.lambda_1))

    return
