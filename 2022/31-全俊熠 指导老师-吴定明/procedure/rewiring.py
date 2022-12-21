import graph
from collections import defaultdict
import random
import math


# 判断两条边是否可以修改
def rewirable(u, v, x, y, k_u, k_v, k_x, k_y):
    # 可以修改的原则：两条边的四个顶点不同，边e1与边e2存在一对相同的点
    if u == v:
        return False
    if u == x:
        return False
    if u == y:
        return False
    if v == x:
        return False
    if v == y:
        return False
    if x == y:
        return False
    if k_u == k_x:
        return True
    if k_u == k_y:
        return True
    if k_v == k_x:
        return True
    if k_v == k_y:
        return True

    return False


# 修订后的度的三角形个数变化
def calculate_number_of_triangles_to_add(genG: graph.Graph, node_degree, u, v, y, num_tri_to_add):
    # 遍历节点u的邻居节点
    for w in genG.nlist[u]:
        # w需要满足度大于1并且与u不同的节点
        if node_degree[w] <= 1 or u == w:
            continue

        # 如果跟节点u相连的节点w就是节点v的话，三点肯定不会构成三角形，而如果节点v的度=1的话，说明节点w肯定不与节点v相连，也没有三角形
        if v != w and node_degree[v] > 1:
            # 如果节点w与节点v相连，w出现的次数就是u、w、v三点构成三角形的个数
            t_minus = genG.nlist[v].count(w)
            # 修订两条边（u跟x相连，v跟y相连）后，三角形个数会减少
            num_tri_to_add[node_degree[u]] -= t_minus
            num_tri_to_add[node_degree[v]] -= t_minus
            num_tri_to_add[node_degree[w]] -= t_minus

        # 如果跟节点u相连的节点w就是节点y的话，修订两条边后，节点y、u、w的三角形数并不会发生变化，而如果节点y的度=1的话，说明节点w肯定不会与节点v相连，也没有三角形
        if y != w and node_degree[y] > 1:
            # 如果节点w与节点y相连，w出现的次数就是u、w、y三点构成三角形的个数
            t_plus = genG.nlist[y].count(w)
            # 修订后，u、w、y三点的三角形个数增加
            num_tri_to_add[node_degree[u]] += t_plus
            num_tri_to_add[node_degree[y]] += t_plus
            num_tri_to_add[node_degree[w]] += t_plus

    return num_tri_to_add


# 随机选择一对边进行修改
def rewiring_random_edge_pair_preserving_joint_degree_matrix(genG: graph.Graph, node_degree, rewirable_edges):
    # 随机选择一条将要修订的边的index
    i_e1 = random.randrange(0, len(rewirable_edges))
    i_e2 = random.randrange(0, len(rewirable_edges))
    # 随机选择的边e1
    e1 = rewirable_edges[i_e1]
    # 随机选择的边e2
    e2 = rewirable_edges[i_e2]

    u = e1[0]   # 边e1的一个端点u
    v = e1[1]  # 边e1的另一个端点v
    x = e2[0]
    y = e2[1]

    k_u = node_degree[u]  # 节点u的度
    k_v = node_degree[v]
    k_x = node_degree[x]
    k_y = node_degree[y]

    # 如果不可以修订，就一直找到一对可以修订的两条边
    while not rewirable(u, v, x, y, k_u, k_v, k_x, k_y):
        i_e1 = random.randrange(0, len(rewirable_edges))
        i_e2 = random.randrange(0, len(rewirable_edges))
        e1 = rewirable_edges[i_e1]
        e2 = rewirable_edges[i_e2]

        u = e1[0]
        v = e1[1]
        x = e2[0]
        y = e2[1]

        k_u = node_degree[u]
        k_v = node_degree[v]
        k_x = node_degree[x]
        k_y = node_degree[y]

    # 两条边修订后，相应度的三角形数量更新
    num_tri_to_add = defaultdict(int)
    # 控制情况
    rewiring_case = -1

    # 如果节点u的度与节点x的度相同或者节点v的度与节点y的度相同
    if k_u == k_x or k_v == k_y:
        # 修订后的度的三角形个数变化
        num_tri_to_add = calculate_number_of_triangles_to_add(genG, node_degree, u, v, y, num_tri_to_add)
        num_tri_to_add = calculate_number_of_triangles_to_add(genG, node_degree, x, y, v, num_tri_to_add)

        # 如果节点v的度大于1并且节点y的度也大于1，节点u、v、y才有可能有三角形
        if node_degree[v] > 1 and node_degree[y] > 1:
            # 三角形数量
            t_minus = genG.nlist[v].count(y)
            # 修订后三角形数量减少
            num_tri_to_add[node_degree[u]] -= t_minus
            # 两倍是因为修订后节点v与y相连
            num_tri_to_add[node_degree[v]] -= 2 * t_minus
            num_tri_to_add[node_degree[x]] -= t_minus
            num_tri_to_add[node_degree[y]] -= 2 * t_minus

        # 另一侧
        if node_degree[u] > 1 and node_degree[x] > 1:
            t_minus = genG.nlist[x].count(u)
            if node_degree[v] > 1:
                num_tri_to_add[node_degree[u]] -= t_minus
                num_tri_to_add[node_degree[v]] -= t_minus
                num_tri_to_add[node_degree[x]] -= t_minus
            if node_degree[y] > 1:
                num_tri_to_add[node_degree[u]] -= t_minus
                num_tri_to_add[node_degree[x]] -= t_minus
                num_tri_to_add[node_degree[y]] -= t_minus

        rewiring_case = 0

    elif k_u == k_y or k_v == k_x:
        num_tri_to_add = calculate_number_of_triangles_to_add(genG, node_degree, u, v, x, num_tri_to_add)
        num_tri_to_add = calculate_number_of_triangles_to_add(genG, node_degree, y, x, v, num_tri_to_add)

        if node_degree[v] > 1 and node_degree[x] > 1:
            t_minus = genG.nlist[v].count(x)
            num_tri_to_add[node_degree[u]] -= t_minus
            num_tri_to_add[node_degree[v]] -= 2 * t_minus
            num_tri_to_add[node_degree[x]] -= 2 * t_minus
            num_tri_to_add[node_degree[y]] -= t_minus

        if node_degree[u] > 1 and node_degree[y] > 1:
            t_minus = genG.nlist[y].count(u)
            if node_degree[v] > 1:
                num_tri_to_add[node_degree[u]] -= t_minus
                num_tri_to_add[node_degree[v]] -= t_minus
                num_tri_to_add[node_degree[y]] -= t_minus
            if node_degree[x] > 1:
                num_tri_to_add[node_degree[u]] -= t_minus
                num_tri_to_add[node_degree[x]] -= t_minus
                num_tri_to_add[node_degree[y]] -= t_minus

        rewiring_case = 1

    return [num_tri_to_add, i_e1, i_e2, rewiring_case]


# 计算当前度依赖聚类系数和目标度依赖聚类系数之间的差距
def calc_L1_distance(tgt_ddcc, cur_ddcc):
    # 出现的度的并集
    degrees = set(tgt_ddcc.keys()) | set(cur_ddcc.keys())
    distance = 0
    # 目标度依赖聚类系数值的总和
    normalization = sum(list(tgt_ddcc.values()))

    # 相同度的距离和
    for d in degrees:
        distance += math.fabs(tgt_ddcc[d] - cur_ddcc[d])

    return [distance, normalization]


# 调整边的连接，使得更接近度依赖聚类系数的估计
def targeting_rewiring_for_clustering(genG: graph.Graph, tgt_ddcc, rewirable_edges, R_C=500):
    # 存放节点度（度要大于1）的字典:{节点v：节点v的度}
    node_degree = defaultdict(int)
    # 存放度k的数量:{度k：数量n}
    degree_number = defaultdict(int)
    # 遍历所有生成图中的所有节点
    for v in genG.nodes:
        # 节点v的度
        degree = len(genG.nlist[v])
        # 如果度大于1
        if degree > 1:
            node_degree[v] = degree
            degree_number[degree] += 1

    # 存放聚类系数相关的常量 2/[d*(d-1)] / n_d
    const_coeff = defaultdict(float)

    for degree in degree_number:
        # 度大于1才会有三角形（聚类系数）
        if degree > 1:
            const_coeff[degree] = float(2) / (degree * (degree - 1))
            const_coeff[degree] = float(const_coeff[degree]) / degree_number[degree]

    # 计算图的度依赖聚类系数和平均聚类系数
    graph.calc_clustering(genG)
    # 当前生成图的度依赖聚类系数的副本
    cur_ddcc = genG.ddcc.copy()

    # 计算当前的度依赖聚类系数的差距
    [distance, normalization] = calc_L1_distance(tgt_ddcc, cur_ddcc)

    # 调整次数
    R = R_C * len(rewirable_edges)

    for r in range(0, R):
        # 当前度依赖聚类系数的副本
        rewired_ddcc = cur_ddcc.copy()
        # 上个阶段修订完后的差距
        rewired_distance = distance

        # 修订边，同时保持联合度矩阵属性，num_tri_to_add 修订后度三角形数量变化，rewiring_case用于判断那种修订
        [num_tri_to_add, i_e1, i_e2, rewiring_case] = rewiring_random_edge_pair_preserving_joint_degree_matrix(genG, node_degree, rewirable_edges)

        # 重新计算生成图的聚类系数
        for degree in num_tri_to_add:
            # 度小于等一1肯定不会有三角形
            if degree > 1:
                # 修改当前度依赖聚类系数的副本
                rewired_ddcc[degree] += float(num_tri_to_add[degree] * const_coeff[degree])
                # 修订后的差距
                rewired_distance += math.fabs(tgt_ddcc[degree] - rewired_ddcc[degree]) - math.fabs(tgt_ddcc[degree] - cur_ddcc[degree])

        # 修订后的差距与修订前的差距的变化
        delta_dist = rewired_distance - distance

        # 变化没有变小，此次修订失效
        if delta_dist >= 0:
            continue

        # 以下是修订成功后生成图更新相应数据
        u = rewirable_edges[i_e1][0]
        v = rewirable_edges[i_e1][1]
        x = rewirable_edges[i_e2][0]
        y = rewirable_edges[i_e2][1]

        # (u,v),(x,y) 换 (u,y),(v,x)
        if rewiring_case == 0:
            graph.remove_edge(genG, u, v)
            graph.remove_edge(genG, x, y)
            graph.add_edge(genG, u, y)
            graph.add_edge(genG, v, x)

            tmp = rewirable_edges[i_e1][1]
            rewirable_edges[i_e1][1] = rewirable_edges[i_e2][1]
            rewirable_edges[i_e2][1] = tmp


        # (u,v),(x,y) 换 (u,x),(v,y)
        elif rewiring_case == 1:
            graph.remove_edge(genG, u, v)
            graph.remove_edge(genG, x, y)
            graph.add_edge(genG, u, x)
            graph.add_edge(genG, v, y)

            tmp = rewirable_edges[i_e1][1]
            rewirable_edges[i_e1][1] = rewirable_edges[i_e2][0]
            rewirable_edges[i_e2][0] = tmp

        cur_ddcc = rewired_ddcc.copy()
        distance = rewired_distance

    return genG