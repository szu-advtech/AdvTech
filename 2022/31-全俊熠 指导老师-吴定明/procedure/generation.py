import numpy as np
from collections import defaultdict
import math
from functools import cmp_to_key
import random
import graph
import estimation
import rewiring


# 比较函数，先按第二个参数排序，小的在前，再按第一个参数排序，大的在前
def cmp(a: list, b: list):
    if a[1] < b[1]:
        return -1
    elif a[1] > b[1]:
        return 1
    else:
        if a[0] < b[0]:
            return 1
        else:
            return -1


# 选择字典中value最小的，如果有多个相同的最小value，则随机选择一个key
def select_random_key_with_smallest_value(dic):
    min_value = float("inf")
    min_value_keys = set()
    for key in dic.keys():
        if dic[key] < min_value:
            min_value = dic[key]
            min_value_keys = set()
            min_value_keys.add(key)
        elif dic[key] == min_value:
            min_value_keys.add(key)
    # 随机选择一个key
    return np.random.choice(list(min_value_keys))


# 选择字典中value最小的，如果有多个相同的最小value，则返回key最小的
def select_min_key_with_smallest_value(dic):
    # 初始一个非常大的值
    min_value = float("inf")
    # 保存最小value的所有key
    min_value_keys = set()
    for key in dic.keys():
        # 小于最小值
        if dic[key] < min_value:
            # 更新最小值
            min_value = dic[key]
            # 清空keys集合
            min_value_keys = set()
            # 添加这个key
            min_value_keys.add(key)
        # 相同最小值error
        elif dic[key] == min_value:
            min_value_keys.add(key)
    # 选择key最小的
    return min(min_value_keys)


# 初始化目标度向量
def initialize_target_degree_vector(est_n, est_dd):
    # 满足 DV-1 度为k的节点目标数量是一个非负整数

    # 初始化目标度向量
    tgt_degree_vector = defaultdict(int)
    for target_degree in est_dd:
        # 目标度数量=度分布估计*节点数估计
        tgt_degree_vector[target_degree] = max(round(est_dd[target_degree] * est_n), 1)

    return tgt_degree_vector


# 调整目标度向量
def adjust_target_degree_vector(est_n, est_dd, tgt_degree_vector):
    # 满足 DV-2 边数应该是偶数

    # 求边的总数
    sum_deg = sum([k * tgt_degree_vector[k] for k in tgt_degree_vector])
    # 如果已经满足 DV-2 就不需要再调整了
    if sum_deg % 2 == 0:
        return tgt_degree_vector

    degree_candidates = {}
    for k in tgt_degree_vector.keys():
        # 选择需要调整的度应该为奇数
        if k % 2 == 0:
            continue
        # 度为k的节点数（估计值）
        est_k_num = est_dd[k] * est_n
        # 度为k的节点数（目标值）
        tgt_k_num = tgt_degree_vector[k]

        # 度为k的概率如果不为0，则计算提升度为k的节点数量之后的error
        if est_k_num != 0:
            delta_e = float(math.fabs(est_k_num - tgt_k_num - 1)) / est_k_num - float(math.fabs(est_k_num - tgt_k_num)) / est_k_num
        # 度为k的概率如果不为0，error无穷大
        else:
            delta_e = float("inf")

        # 提升度为k的节点数的候选这的误差
        degree_candidates[k] = delta_e

    # 存在可以提升度为k的节点数量的候选者
    if len(degree_candidates) > 0:
        # 选择error最小的候选者度k
        condidate_degree = select_min_key_with_smallest_value(degree_candidates)
        # 度为k的目标数量加一
        tgt_degree_vector[condidate_degree] += 1
    else:
        # 不存在奇数的候选者，则就让度为1的节点数增加一
        tgt_degree_vector[1] += 1

    return tgt_degree_vector


# 修改目标度向量
def modify_target_degree_vector(subG: graph.Graph, est_n, est_dd, tgt_degree_vector):
    # 分配给每个节点目标度
    # 满足 DV-3 目标度向量不小于子图的度向量

    # 子图（生成图）的度向量
    subG_degree_vector = defaultdict(int)
    # 存放目标节点的度
    tgt_node_degree = {}
    # 遍历查询节点集合
    for v in subG.qry_nodes:
        # 节点v在子图中的度
        subG_degree = len(subG.nlist[v])
        # {节点：度}
        tgt_node_degree[v] = subG_degree
        # 更新子图的度向量
        subG_degree_vector[subG_degree] += 1

    # 调整查询节点的度的节点数量
    for degree in subG_degree_vector:
        # 如果查询节点的度的目标数量小于子图（加入查询节点）中该度的数量
        if tgt_degree_vector[degree] < subG_degree_vector[degree]:
            # 增大目标度向量到与子图该度相同的数量
            tgt_degree_vector[degree] = subG_degree_vector[degree]

    # [[邻居可访问节点，该节点的度],....]
    visible_node_pairs = []
    # 遍历邻居可访问节点集合
    for v in subG.vis_nodes:
        visible_node_pairs.append([v, len(subG.nlist[v])])
    # 排序，度小的在前面，度相同时，节点index大的在前
    visible_node_pairs.sort(key=cmp_to_key(cmp))

    for visible_node_pair in visible_node_pairs:
        # 节点v
        v = visible_node_pair[0]
        # 节点v在子图（当前生成图）中的度
        subG_degree = visible_node_pair[1]

        # 候选度，当前还可以分配的度
        degree_candidates = []
        # 统计还可以分配给节点的度
        for k in tgt_degree_vector:
            # 从目标度向量中找到一个不小于节点v的度的度k，并且这个度k的目标节点数量要大于子图（当前生成图）这个度k的节点数量
            if k >= subG_degree and tgt_degree_vector[k] > subG_degree_vector[k]:
                # 可以存入（度k的目标数量-当前度k的数量）个度k到候选度list中
                for i in range(0, tgt_degree_vector[k] - subG_degree_vector[k]):
                    degree_candidates.append(k)

        # 存在可分配的度
        if len(degree_candidates) > 0:
            # 从候选度list中随机选择一个度赋值给节点v
            tgt_node_degree[v] = np.random.choice(list(degree_candidates))
        # 不存在可分配的度
        else:
            # {度k：error}
            degree_to_add_candidates = {}
            # 从度分布估计中找
            for k in est_dd:
                # 找不小于节点v的度的度k
                if k < subG_degree:
                    continue
                # 度k的节点数量的估计
                est_k_num = est_dd[k] * est_n
                # 度k的目标数量
                tgt_k_num = float(tgt_degree_vector[k])
                # 增加度k的目标数量之后的error
                if est_k_num != 0:
                    delta_e = float(math.fabs(est_k_num - tgt_k_num - 1)) / est_k_num - float(math.fabs(est_k_num - tgt_k_num)) / est_k_num
                else:
                    delta_e = float("inf")
                # 添加到候选度中
                degree_to_add_candidates[k] = delta_e

            # 存在可增加数量的度k
            if len(degree_to_add_candidates) > 0:
                # 选择增加度k节点数量之后error最小的度k，分配给节点v
                tgt_node_degree[v] = select_min_key_with_smallest_value(degree_to_add_candidates)
            # 不存在
            else:
                # 保持不变，节点v的度为子图（当前生成图）中的度
                tgt_node_degree[v] = subG_degree

            # 增加目标度k的节点数量+1
            tgt_degree_vector[tgt_node_degree[v]] += 1
        # 再+1
        subG_degree_vector[tgt_node_degree[v]] += 1

    # 改变目标度向量之后，需要对目标度向量进行重新调整
    tgt_degree_vector = adjust_target_degree_vector(est_n, est_dd, tgt_degree_vector)

    return [subG_degree_vector, tgt_degree_vector, tgt_node_degree]


# 初始化目标联合度矩阵
def initialize_target_joint_degree_matrix(est_n, est_aved, est_jdd):
    # 满足 JDM-1 和 JDM-2 两个条件
    # JDM-1：联合度矩阵中的元素是非负整数
    # JDM-2：m(k1,k2) = m(k2,k1) 对任意k1不等于k2

    # 初始化目标联合度分布
    tgt_joint_degree_matrix = defaultdict(lambda: defaultdict(int))
    # 遍历联合度分布的估计矩阵
    for k1 in est_jdd:
        for k2 in est_jdd[k1]:
            # 如果联合度小于等于0，跳过
            if est_jdd[k1][k2] <= 0:
                continue

            # 节点数估计*平均度估计*度为k1和度为k2之间的边数估计
            k1_k2_edges_num = round(est_n * est_aved * est_jdd[k1][k2])

            # 当两个度不相等的时候
            if k1 != k2:
                # 度为k1和k2之间联合度初始化
                tgt_joint_degree_matrix[k1][k2] = max(k1_k2_edges_num, 1)
            # 当两个度相等的时候
            else:
                # x已经是偶数时，可以直接赋值
                if k1_k2_edges_num % 2 == 0:
                    tgt_joint_degree_matrix[k1][k2] = max(k1_k2_edges_num, 2)
                # x为奇数时
                else:
                    # 边数估计原始值，未四舍五入
                    k1_k2_edges_num_raw = est_n * est_aved * est_jdd[k1][k2]
                    # y离x-1更近，说明x是四舍
                    if math.fabs(k1_k2_edges_num_raw - k1_k2_edges_num + 1) <= math.fabs(k1_k2_edges_num_raw - k1_k2_edges_num - 1):
                        tgt_joint_degree_matrix[k1][k2] = max(k1_k2_edges_num - 1, 2)
                    # y离x+1更近，说明x是五入
                    else:
                        tgt_joint_degree_matrix[k1][k2] = max(k1_k2_edges_num + 1, 2)

    return tgt_joint_degree_matrix


# 调整目标联合度矩阵
def adjust_target_joint_degree_matrix(est_n, est_aved, est_jdd, tgt_degree_vector, min_joint_degree_matrix, tgt_joint_degree_matrix):
    # 满足 JDM-3：目标联合度矩阵中所有度为k相关的边数总和=k*目标度k的数量

    # 目标度向量中值不为0的key集合
    degree_k1_set = set(tgt_degree_vector.keys())
    # 如果度为1不存在，则添加到key集合中
    if 1 not in degree_k1_set:
        degree_k1_set.add(1)
    # 对目标度向量中的key（度）进行排序，度大的放在前面
    degree_k1_set = sorted(list(degree_k1_set), reverse=True)

    # 从大到小遍历度k1
    for k1 in degree_k1_set:
        # 目标边总数，JDM-3 等式右侧
        target_sum = k1 * tgt_degree_vector[k1]
        # 当前边总数，JDM-3 等式左侧
        present_sum = sum(tgt_joint_degree_matrix[k1].values())
        # 差距
        different = target_sum - present_sum

        # JDM-3 条件满足，则无需调整
        if different == 0:
            continue

        # 小于等于度k1的度k2集合
        degree_k2_set = set([k2 for k2 in degree_k1_set if k2 <= k1])

        # 如果度k1是1，并且差距是奇数，就没有办法通过有限调整使得JDM-3成立
        if k1 == 1 and abs(target_sum - present_sum) % 2 != 0:
            # 度为1的目标向量增加一
            tgt_degree_vector[1] += 1
            # 目标总数+1
            target_sum += 1

        # 一直调整到JDM-3成立
        while target_sum != present_sum:
            # 如果当前比目标小，需要增大当前边的总数
            if target_sum > present_sum:
                # 可以用来调整的k2集合
                degree_k2_candidate = {}
                for k2 in degree_k2_set:
                    # 如果差距为一，没有办法通过调整m(k,k)使差距为0
                    if present_sum == target_sum - 1 and k2 == k1:
                        continue

                    # 联合度k的估计，k‘之间的边数=联合度分布估计*节点总数估计*平均度估计
                    est_k1_k2_edges_num = est_jdd[k1][k2] * est_n * est_aved
                    tgt_k1_k2_edges_num = float(tgt_joint_degree_matrix[k1][k2])

                    # 联合度k，k’概率估计为0
                    if est_k1_k2_edges_num == 0:
                        # error设为无穷
                        delta_e = float("inf")
                    else:
                        # 如果不是同度，计算将目标联合度m(k1,k2)增加1后的误差
                        if k2 != k1:
                            delta_e = float(math.fabs(est_k1_k2_edges_num - tgt_k1_k2_edges_num - 1)) / est_k1_k2_edges_num - float(math.fabs(est_k1_k2_edges_num - tgt_k1_k2_edges_num)) / est_k1_k2_edges_num
                        # 如果同度，计算将目标联合度m(k,k)增加2后的误差
                        else:
                            delta_e = float(math.fabs(est_k1_k2_edges_num - tgt_k1_k2_edges_num - 2)) / est_k1_k2_edges_num - float(math.fabs(est_k1_k2_edges_num - tgt_k1_k2_edges_num)) / est_k1_k2_edges_num
                    # 保存修改k1与k2的目标联合度的误差，k2候选集合
                    degree_k2_candidate[k2] = delta_e

                # 选择一个误差最小的k2，如果存在多个误差相同的k2，就随机选择一个k2
                k2 = select_random_key_with_smallest_value(degree_k2_candidate)

                # 修改相应的目标联合度矩阵
                tgt_joint_degree_matrix[k1][k2] += 1
                tgt_joint_degree_matrix[k2][k1] += 1

                # 修改当前边总数，如果k2与k1相同边数是增加2，不相同是增加1
                if k1 != k2:
                    present_sum += 1
                else:
                    present_sum += 2

            # 如果当前比目标大，需要减小当前边的总数
            else:
                degree_k2_candidate = {}
                for k2 in degree_k2_set:
                    # 因为需要降低目标联合度，所以可能会发生降低后的目标联合度低于下限
                    if tgt_joint_degree_matrix[k1][k2] <= min_joint_degree_matrix[k1][k2]:
                        continue
                    if present_sum == target_sum + 1 and k2 == k1:
                        continue

                    est_k1_k2_edges_num = est_jdd[k1][k2] * est_n * est_aved
                    tgt_k1_k2_edges_num = float(tgt_joint_degree_matrix[k1][k2])

                    if est_k1_k2_edges_num == 0:
                        delta_e = float("inf")
                    else:
                        if k2 != k1:
                            delta_e = float(math.fabs(est_k1_k2_edges_num - tgt_k1_k2_edges_num + 1)) / est_k1_k2_edges_num - float(math.fabs(est_k1_k2_edges_num - tgt_k1_k2_edges_num)) / est_k1_k2_edges_num
                        else:
                            delta_e = float(math.fabs(est_k1_k2_edges_num - tgt_k1_k2_edges_num + 2)) / est_k1_k2_edges_num - float(math.fabs(est_k1_k2_edges_num - tgt_k1_k2_edges_num)) / est_k1_k2_edges_num

                    degree_k2_candidate[k2] = delta_e

                if len(degree_k2_candidate) > 0:
                    k2 = select_random_key_with_smallest_value(degree_k2_candidate)
                    tgt_joint_degree_matrix[k1][k2] -= 1
                    tgt_joint_degree_matrix[k2][k1] -= 1

                    if k1 != k2:
                        present_sum -= 1
                    else:
                        present_sum -= 2
                # 如果不存在可以调整的k2
                else:
                    if k1 > 1:
                        target_sum += k1
                        tgt_degree_vector[k1] += 1
                    else:
                        target_sum += 2
                        tgt_degree_vector[1] += 2

    return [tgt_joint_degree_matrix, tgt_degree_vector]


# 修改目标联合度矩阵
def modify_target_joint_degree_matrix(subG: graph.Graph, est_n, est_aved, est_jdd, tgt_node_degree, tgt_degree_vector, tgt_joint_degree_matrix):
    # 满足 JDM-4：目标联合度都要不小于对应的估计值

    # 目标度向量中值不为0的key集合
    degree_set = set(tgt_degree_vector.keys())
    # 保证度集合中包含度为1
    if 1 not in degree_set:
        degree_set.add(1)
    # 对度k从小到大排列
    degree_set = set(sorted(list(degree_set)))

    # 初始化当前子图的联合度矩阵
    subG_joint_degree_matrix = defaultdict(lambda: defaultdict(int))
    # 遍历子图的节点集，得到当前子图的联合度矩阵
    for v in subG.nodes:
        # 节点v的目标度
        k1 = tgt_node_degree[v]
        # 遍历节点v的邻居
        for w in subG.nlist[v]:
            # 遍历节点v的邻居
            k2 = tgt_node_degree[w]
            # 度为k1的顶点v和度为k2的顶点w之间存在一条边
            subG_joint_degree_matrix[k1][k2] += 1


    # 遍历子图的联合度矩阵
    for k1 in subG_joint_degree_matrix:
        for k2 in subG_joint_degree_matrix[k1]:
            # 不满足 JDM-4 的时候
            while subG_joint_degree_matrix[k1][k2] > tgt_joint_degree_matrix[k1][k2]:
                # 目标联合度m(k1,k2)、m(k2,k1)都增加一
                tgt_joint_degree_matrix[k1][k2] += 1
                tgt_joint_degree_matrix[k2][k1] += 1

                # k3候选集，因为k2的联合度变大了，所以需要找一个k3使k2联合度变小
                degree_k3_candidates = {}
                for k3 in degree_set:
                    # 找到一个不等于k1的k3，并且m(k2,k3)大于当前子图对应的联合度
                    if k3 == k1 or tgt_joint_degree_matrix[k2][k3] <= subG_joint_degree_matrix.get(k2, {}).get(k3, 0):
                        continue

                    est_k2_k3_edges_num = est_jdd[k2][k3] * est_n * est_aved
                    tgt_k2_k3_edges_num = tgt_joint_degree_matrix[k2][k3]

                    if est_k2_k3_edges_num == 0:
                        delta_e = float("inf")
                    else:
                        if k2 != k3:
                            delta_e = float(math.fabs(est_k2_k3_edges_num - tgt_k2_k3_edges_num + 1)) / est_k2_k3_edges_num - float(math.fabs(est_k2_k3_edges_num - tgt_k2_k3_edges_num)) / est_k2_k3_edges_num
                        else:
                            delta_e = float(math.fabs(est_k2_k3_edges_num - tgt_k2_k3_edges_num + 2)) / est_k2_k3_edges_num - float(math.fabs(est_k2_k3_edges_num - tgt_k2_k3_edges_num)) / est_k2_k3_edges_num

                    degree_k3_candidates[k3] = delta_e

                k3 = -1
                # 如果存在k3，选择一个error最小的k3进行修改
                if len(degree_k3_candidates) > 0:
                    k3 = select_random_key_with_smallest_value(degree_k3_candidates)
                    tgt_joint_degree_matrix[k2][k3] -= 1
                    tgt_joint_degree_matrix[k3][k2] -= 1

                # k2的联合度虽然通过k3使其保持不变，但是k3的联合度变大了，此时k1联合度也是变大的
                degree_k4_candidates = {}
                for k4 in degree_set:
                    if k4 == k2 or tgt_joint_degree_matrix[k1][k4] <= subG_joint_degree_matrix.get(k1, {}).get(k4, 0):
                        continue

                    est_k1_k4_edges_num = est_jdd[k1][k4] * est_n * est_aved
                    tgt_k1_k4_edges_num = tgt_joint_degree_matrix[k1][k4]

                    if est_k1_k4_edges_num == 0:
                        delta_e = float("inf")
                    else:
                        if k1 != k4:
                            delta_e = float(math.fabs(est_k1_k4_edges_num - tgt_k1_k4_edges_num + 1)) / est_k1_k4_edges_num - float(math.fabs(est_k1_k4_edges_num - tgt_k1_k4_edges_num)) / est_k1_k4_edges_num
                        else:
                            delta_e = float(math.fabs(est_k1_k4_edges_num - tgt_k1_k4_edges_num + 2)) / est_k1_k4_edges_num - float(math.fabs(est_k1_k4_edges_num - tgt_k1_k4_edges_num)) / est_k1_k4_edges_num

                    degree_k3_candidates[k4] = delta_e

                # 如果存在k4，选择一个error最小的k3进行修改
                if len(degree_k4_candidates) > 0:
                    k4 = select_random_key_with_smallest_value(degree_k4_candidates)
                    tgt_joint_degree_matrix[k4][k1] -= 1
                    tgt_joint_degree_matrix[k1][k4] -= 1

                    if k3 > 0:
                        tgt_joint_degree_matrix[k3][k4] += 1
                        tgt_joint_degree_matrix[k4][k3] += 1

    # 修改目标联合度矩阵之后，可能会打破JDM-1，JDK-2的条件，所以需要重新调整一次
    [tgt_jnt_degree_matrix, tgt_degree_vector] = adjust_target_joint_degree_matrix(est_n, est_aved, est_jdd, tgt_degree_vector, subG_joint_degree_matrix, tgt_joint_degree_matrix)

    return [subG_joint_degree_matrix, tgt_jnt_degree_matrix, tgt_degree_vector]


# 构建子图
def construct_subG(genG: graph.Graph, sampling_list):
    # {节点：index}
    node_index = {}
    edges_to_add = []

    # 用index表示查询节点集合和边集合
    i = 0
    for data in sampling_list:
        v = data.index
        if v not in node_index:
            node_index[v] = i
            i += 1
    # index在0到当前i之间的节点属于访问节点
    genG.qry_nodes = set(range(0, i))

    # 用index表示可访问节点集合和边集合
    # 存储已经访问的节点，用于不重复添加邻居节点
    marked = set()
    for data in sampling_list:
        v = data.index
        if v in marked:
            continue

        marked.add(v)
        for w in data.nlist:
            # 还未添加到子图的节点集合当中
            if w not in node_index:
                node_index[w] = i
                i += 1
            # 节点v与节点w之间的边还未添加到edges_to_add边集中
            if w not in marked:
                # 添加的边用节点index表示
                edges_to_add.append([node_index[v], node_index[w]])

    # index在len(genG.qry_nodes)到当前i之间的节点属于邻居可访问节点
    genG.vis_nodes = set(range(len(genG.qry_nodes), i))

    # 构建子图，将节点集合与边集合保存到Graph对象当中
    # 集合并集
    genG.nodes = genG.qry_nodes | genG.vis_nodes
    # 添加边
    for [v, w] in edges_to_add:
        graph.add_edge(genG, v, w)

    return genG


# 原图属性估计
def original_properties_estimation(sampling_list):
    # 节点总数估计
    est_n = estimation.size_estimator(sampling_list)
    # 平均度估计
    est_aved = estimation.average_degree_estimator(sampling_list)
    # 度分布估计：P(k) 度为k的节点出现的概率   字典类型：key为某个度数k，value表示P(k)
    est_dd = estimation.degree_distribution_estimator(sampling_list)
    # 联合度分布估计（混合方法）   {度k：{度l：度为k的节点与度为l的节点之间的边数,....},....}
    est_jdd = estimation.JDD_estimator_hybrid(sampling_list, est_n, est_aved)
    # 度依赖聚类系数的估计
    est_ddcc = estimation.degree_dependent_clustering_coefficient_estimator(sampling_list)
    return [est_n, est_aved, est_dd, est_jdd, est_ddcc]


# 构建目标度向量
def construct_tgt_degree_vector(genG: graph.Graph, est_n, est_dd):
    # 步骤一：初始化目标度向量    字典类型：key为某个度数k，value表示度为k的节点数量
    tgt_degree_vector = initialize_target_degree_vector(est_n, est_dd)
    # 步骤二：调整目标度向量
    tgt_degree_vector = adjust_target_degree_vector(est_n, est_dd, tgt_degree_vector)
    # 步骤三：修改目标度向量
    [subG_degree_vector, tgt_degree_vector, tgt_node_degree] = modify_target_degree_vector(genG, est_n, est_dd, tgt_degree_vector)
    return [subG_degree_vector, tgt_degree_vector, tgt_node_degree]


# 构建目标联合度矩阵
def construct_tgt_joint_degree_matrix(genG: graph.Graph, est_n, est_aved, est_jdd, tgt_degree_vector, tgt_node_degree):
    # 步骤一：初始化目标联合度矩阵
    tgt_joint_degree_matrix = initialize_target_joint_degree_matrix(est_n, est_aved, est_jdd)
    # 最小联合度矩阵，用于调整目标联合度矩阵
    min_joint_degree_matrix = defaultdict(lambda: defaultdict(int))
    # 步骤二：调整目标联合度矩阵
    [tgt_joint_degree_matrix, tgt_degree_vector] = adjust_target_joint_degree_matrix(est_n, est_aved, est_jdd,
                                                                                     tgt_degree_vector,
                                                                                     min_joint_degree_matrix,
                                                                                     tgt_joint_degree_matrix)
    # 步骤三：修改目标联合度矩阵
    [subG_joint_degree_matrix, tgt_joint_degree_matrix, tgt_degree_vector] = modify_target_joint_degree_matrix(genG,
                                                                                                               est_n,
                                                                                                               est_aved,
                                                                                                               est_jdd,
                                                                                                               tgt_node_degree,
                                                                                                               tgt_degree_vector,
                                                                                                               tgt_joint_degree_matrix)
    return [subG_joint_degree_matrix, tgt_joint_degree_matrix, tgt_degree_vector]


# 对子图进行构造，使得结构满足目标度向量和目标联合度矩阵
def reconstruct_genG_with_two_tgt(genG: graph.Graph, tgt_degree_vector, subG_degree_vector, tgt_node_degree, tgt_joint_degree_matrix, subG_joint_degree_matrix):
    # (4-1) 决定目标节点数量
    # 目标节点数
    tgt_N = sum(list(tgt_degree_vector.values()))
    # 当前子图的节点数
    subG_N = len(genG.nodes)
    # 添加节点，使当前子图的节点满足目标节点数
    for v in range(subG_N, tgt_N):
        genG.nodes.add(v)

    # (4-2) 给添加进来的节点分配目标度
    # 还可以分配的度
    degree_seq = []
    for d in tgt_degree_vector:
        for i in range(0, tgt_degree_vector[d] - subG_degree_vector[d]):
            degree_seq.append(d)

    # 当前子图的度向量副本
    cur_degree_vector = defaultdict(int)
    for d in tgt_degree_vector:
        cur_degree_vector[d] = subG_degree_vector[d]

    # 打乱可以分配的度
    random.shuffle(degree_seq)
    # 给刚添加的节点分配度
    for v in range(subG_N, tgt_N):
        # 取出可分配的第一个度d
        d = degree_seq.pop()
        # 分配给节点v
        tgt_node_degree[v] = d
        # 当前度d数量加一
        cur_degree_vector[d] += 1

    # (4-3) 统计自由边
    # 存放度为k的节点v还可以连出多少条边（自由边）
    free_edges = defaultdict(list)
    for v in genG.nodes:
        # 节点v的度
        tgt_degree = tgt_node_degree[v]
        # 节点v当前子图中的度
        subG_degree = len(genG.nlist[v])
        for i in range(0, tgt_degree - subG_degree):
            free_edges[tgt_degree].append(v)

    # 打乱自由边
    for degree in free_edges:
        random.shuffle(free_edges[degree])

    # (4-4) 随机连接度为k1的节点和度为k2的节点的自由边
    # 存档当前子图的联合度矩阵副本
    cur_joint_degree_matrix = defaultdict(lambda: defaultdict(int))
    for k1 in tgt_joint_degree_matrix:
        for k2 in tgt_joint_degree_matrix[k1]:
            cur_joint_degree_matrix[k1][k2] = subG_joint_degree_matrix[k1][k2]

    for k1 in tgt_joint_degree_matrix:
        for k2 in tgt_joint_degree_matrix[k1]:
            # 当前子图的联合度矩阵与目标联合度矩阵不等时
            while cur_joint_degree_matrix[k1][k2] != tgt_joint_degree_matrix[k1][k2]:
                # 从度为k1的自由边集合中选择一条边u
                u = free_edges[k1].pop()
                # 从度为k2的自由边集合中选择一条边v
                v = free_edges[k2].pop()
                # 将节点u与节点v加入到子图中
                graph.add_edge(genG, u, v)
                # 更新当前联合度矩阵
                cur_joint_degree_matrix[k1][k2] += 1
                cur_joint_degree_matrix[k2][k1] += 1

    return genG


# 图生成方法
def graph_restoration_method(sampling_list):
    # 初始化存放生成图的Graph对象
    genG = graph.Graph()

    # (1) 构建子图
    genG = construct_subG(genG, sampling_list)

    # 如果没有邻居可访问的节点，就返回子图
    if len(genG.vis_nodes) == 0:
        # 设置节点数
        genG.N = len(genG.nodes)
        # 设置边数
        genG.M = 0
        # 设置最大度
        genG.maxd = 0
        for v in genG.nodes:
            d = len(genG.nlist[v])
            genG.M += d
            if d > genG.maxd:
                # 设置最大度
                genG.maxd = d
        # 设置边数
        genG.M = int(genG.M / 2)

        return genG

    # (2) 原图属性估计
    [est_n, est_aved, est_dd, est_jdd, est_ddcc] = original_properties_estimation(sampling_list)

    # (3) 构建目标度向量
    [subG_degree_vector, tgt_degree_vector, tgt_node_degree] = construct_tgt_degree_vector(genG, est_n, est_dd)

    # (4) 构建目标联合度矩阵
    [subG_joint_degree_matrix, tgt_joint_degree_matrix, tgt_degree_vector] = construct_tgt_joint_degree_matrix(genG, est_n, est_aved, est_jdd, tgt_degree_vector, tgt_node_degree)

    # (5) 对子图进行构造，使得结构满足目标度向量和目标联合度矩阵
    genG = reconstruct_genG_with_two_tgt(genG, tgt_degree_vector, subG_degree_vector, tgt_node_degree, tgt_joint_degree_matrix, subG_joint_degree_matrix)

    # 子图的边数量
    genG.M = 0
    # 子图的最大度
    genG.maxd = 0
    for v in genG.nodes:
        d = len(genG.nlist[v])
        genG.M += d
        if d > genG.maxd:
            genG.maxd = d
    genG.M = int(genG.M / 2)

    # (6) 重新布线
    # 可以修订的边集
    rewirable_edges = []
    # 遍历邻居可访问节点v
    for v in range(len(genG.qry_nodes), len(genG.nodes)):
        for w in genG.nlist[v]:
            # 如果节点w也是邻居可访问节点
            if w >= v and w >= len(genG.qry_nodes):
                rewirable_edges.append([v, w])

    # 根据目标聚类系数进行修改
    genG = rewiring.targeting_rewiring_for_clustering(genG, est_ddcc, rewirable_edges)

    return genG
