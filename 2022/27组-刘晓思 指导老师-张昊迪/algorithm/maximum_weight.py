from common.route import Route
from copy import deepcopy

def maximum_weight(trajs, e, budget, k, lk):

    lookup_table = deepcopy(lk)


    # 基于 lookup table构建基于权重的邻接矩阵
    edges = e[:]
    # 初始化A邻接矩阵
    A_matrix = [[-1 for x in range(len(edges))] for x in range(len(edges))]

    num = 0

    for i in range(len(edges)):
        for j in range(len(edges)):
            if edges[i].end == edges[j].start:
                if edges[i].id in lookup_table.keys():
                    A_matrix[i][j] = len(lookup_table[edges[i].id])
                    num += 1
                else:
                    A_matrix[i][j] = 0

    # root vertex & bottom vertex
    A_matrix[0] = [0 for x in range(len(edges))]
    A_matrix[-1] = [0 for x in range(len(edges))]

    routes = []
    t = []
    r_idx = 0

    # # 初始化mp向量
    # mp = [0 for x in range(len(edges))]
    while len(routes) < k:

        # 初始化mp向量
        mp = [0 for x in range(len(edges))]

        # 路径以及路径的代价

        r = Route(r_idx, None, None, [])
        cr = 0

        for i in range(len(edges)):
            ce = edges[i].length
            for j in range(len(edges)):

                # retrieve the route 𝑟 𝑗 whose last edge is 𝑒𝑗 by 𝑚𝑝 [ 𝑗 ]

                if A_matrix[j][i] != -1 and cr + ce <= budget:
                    # 添加路段到路径中？
                    mp[i] = max(mp[i], mp[j] + A_matrix[j][i])

                    # ???
                    # r.add_edge(edges[i])
                    if r.add_edge(edges[j]):
                        A_matrix[i][j] = -1
                        A_matrix[j][i] = -1

        # retrieve the route 𝑟 by 𝑚𝑝 [ |𝐴 | − 1] and update 𝐴, 𝑅, and T
        # 添加路径
        routes.append(r)
        r_idx += 1

        print('id:{}, length:{}'.format(r_idx, len(r.e_list)))

        # 更新 t
        for e in r.e_list:
            if e.id in lookup_table.keys():
                t = list(set(t + lookup_table[e.id]))
        print('t:{}'.format(len(t)))

    s = len(t) / len(trajs)
    print(s)

    return routes, s