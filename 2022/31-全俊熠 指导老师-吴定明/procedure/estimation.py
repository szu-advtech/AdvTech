from collections import defaultdict


# 基于重新加权随机游走的无偏估计器

def size_estimator(sampling_list):
    # Estimating sizes of social networks via biased sampling, WWW, 2011.

    r = len(sampling_list)
    # parameter
    m = int(round(r * 0.025))
    phi = 0.0
    psi = 0.0
    for k in range(0, r - m):
        # 第k个节点的index
        v_k = sampling_list[k].index
        # 第k个节点的度
        d_k = len(sampling_list[k].nlist)
        for l in range(k + m, r):
            # 第l个节点
            v_l = sampling_list[l].index
            # 第l个节点的度
            d_l = len(sampling_list[l].nlist)
            # 节点k与节点l是同一个节点
            if v_k == v_l:
                # 发生碰撞的节点对，+2是因为分母是2C
                phi += 2
            # 所有采样节点度的和*所有采样节点度的倒数的和
            psi += (float(d_k) / d_l) + (float(d_l) / d_k)

    return float(psi) / phi


def average_degree_estimator(sampling_list):
    # Walking in Facebook: A case study of unbiased sampling of osns, INFOCOM, 2010.

    # 采样子集中每个节点度的倒数的总和
    est = 0.0
    for data in sampling_list:
        est += float(1) / len(data.nlist)
    return float(len(sampling_list)) / est


def degree_distribution_estimator(sampling_list):
    # Walking in Facebook: A case study of unbiased sampling of osns, INFOCOM, 2010.

    # P(k=i) = [度为i的节点度的倒数的总和] / [所有节点度的倒数的总和]
    est = defaultdict(float)
    x = 0
    for data in sampling_list:
        d = len(data.nlist)
        est[d] += float(1) / d
        x += float(1) / d

    for d in est:
        est[d] = float(est[d]) / x

    return est


def jdd_estimator_induced_edges(sampling_list, est_n, est_aved):
    # 2.5 k-graphs: from sampling to generation, INFOCOM, 2013.
    # The original estimator does not correctly converge to the real value.
    # This is a modified unbiased estimator.

    # 存放分子，度为k的所有节点和度为l的所有节点之间的边数占比
    phi = defaultdict(lambda: defaultdict(float))
    r = len(sampling_list)
    # 表示 margin 大小
    m = int(round(r * 0.025))

    # 公式中的分子计算
    for i in range(0, r - m):
        v = sampling_list[i].index
        # 节点i的度
        k = len(sampling_list[i].nlist)
        for j in range(i + m, r):
            # 节点j的度
            l = len(sampling_list[j].nlist)
            # 节点j和节点i之间的边数量
            c = sampling_list[j].nlist.count(v)
            # 度为k的节点i和度为l的节点j之间的边数的占比，（疑问：除以k*l是为了消除bais吗？）
            value = float(c) / (k * l)
            phi[k][l] += value
            phi[l][k] += value

    # 联合度分布的估计
    est = defaultdict(lambda: defaultdict(float))
    # 参加计算的 pairs 总数，公式中的分母
    num_sample = (r - m) * (r - m + 1)
    # 原图节点数的估计值，原图平均度的估计值，用来替换公式中的|Vk|*|Vl|
    sum_d = est_n * est_aved
    for k in phi:
        for l in phi[k]:
            value = float(phi[k][l]) / num_sample
            value *= sum_d
            est[k][l] = value

    return est


def jdd_estimator_traversed_edges(sampling_list):
    # 2.5 k-graphs: from sampling to generation, INFOCOM, 2013.
    # The original estimator does not correctly converge to a real value.
    # This is a modified unbiased estimator.

    est = defaultdict(lambda: defaultdict(float))
    r = len(sampling_list)

    # 按照随机游走采样的顺序
    for i in range(0, r - 1):
        k = len(sampling_list[i].nlist)  # 节点i的度
        l = len(sampling_list[i + 1].nlist)  # 节点i+1的度
        est[k][l] += 1  # 边数自增
        est[l][k] += 1

    for k in est:
        for l in est[k]:
            # 每个节点会重复算两次，所以要除以2，分母是采样边数（随机游走路过的边数，采样节点数-1）
            est[k][l] = float(est[k][l]) / (2 * (r - 1))

    return est


def JDD_estimator_hybrid(sampling_list, est_n, est_aved):
    # 2.5 k-graphs: from sampling to generation, INFOCOM, 2013.
    # The original estimator does not correctly converge to a real value.
    # This is a modified unbiased estimator.

    est_jdd_ie = jdd_estimator_induced_edges(sampling_list, est_n, est_aved)
    est_jdd_te = jdd_estimator_traversed_edges(sampling_list)

    est_jdd = defaultdict(lambda: defaultdict(float))

    for k in est_jdd_ie:
        for l in est_jdd_ie[k]:
            if (k + l) >= 2 * est_aved:
                est_jdd[k][l] = est_jdd_ie[k][l]
                est_jdd[l][k] = est_jdd_ie[l][k]

    for k in est_jdd_te:
        for l in est_jdd_te[k]:
            if (k + l) < 2 * est_aved:
                est_jdd[k][l] = est_jdd_te[k][l]
                est_jdd[l][k] = est_jdd_te[l][k]

    return est_jdd


def degree_dependent_clustering_coefficient_estimator(sampling_list):
    # Estimating clustering coefficients and size of social networks via random walk, WWW, 2013.
    # 度依赖聚类系数(平均聚类系数)：度为k的所有节点的聚类系数总和

    # fai
    phi = defaultdict(float)
    # 坡赛
    psi = defaultdict(float)
    r = len(sampling_list)

    for i in range(0, r):
        # 随机游走采样序列第i个节点的度
        d = len(sampling_list[i].nlist)
        psi[d] += float(1) / d

    # 第i-1个节点与第i+1个节点之间的边数
    for i in range(1, r - 1):
        d = len(sampling_list[i].nlist)
        # 如果第i个节点的度小于2的话，就一定不会有三角形
        if d == 0 or d == 1:
            continue

        # 采样到的第i-1个节点的index
        s = sampling_list[i - 1].index
        # 第i个节点的index
        v = sampling_list[i].index
        # 第i+1个节点的index
        t = sampling_list[i + 1].index

        # 连续采样的三个节点需要不同，才可能存在一这三个节点为顶点的三角形
        if s != v and v != t and t != s:
            # 节点s与节点t之间的边数
            c = sampling_list[i + 1].nlist.count(s)
            # f(v) = 1 / (d - 1)
            phi[d] += float(c) / (d - 1)

    for d in phi:
        phi[d] = float(phi[d]) / (r - 2)

    for d in psi:
        psi[d] = float(psi[d]) / r

    est_ddcc = defaultdict(float)
    for d in psi:
        est_ddcc[d] = float(phi[d]) / psi[d]

    return est_ddcc
