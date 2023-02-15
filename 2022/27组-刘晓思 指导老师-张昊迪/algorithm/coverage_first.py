from common.route import Route
from copy import deepcopy

# 比较两个路径集合是否不相等
def compare_r(r1, r2):
    if len(r1) == 0 or len(r2) == 0:
        return False
    if len(r1) != len(r2):
        return True
    for i in range(len(r1)):
        if len(r1[i].e_list) != len(r2[i].e_list):
            return True
    return False

def coverage_first(trajs, e, budget, k, lk):

    lookup_table = deepcopy(lk)

    # 覆盖优先算法
    edges = e[:]

    # edges = [e[eid] for eid in lookup_table.keys()]

    t = []

    em = []

    route = []

    r_idx = 0

    while len(lookup_table) > 0:

        # print('len edges:{}'.format(len(lookup_table)))

        # 选取the first edge for a route 获取覆盖最多轨迹的边
        # lk_edges = list(set(lookup_table.keys()).difference(set(em)))
        # lk_edges = edges
        # target_e_idx = 0
        # for i in range(len(lk_edges) - 1):
        #     # 因为lookup table排好序，只要这个uncover大小比下一个边覆盖的轨迹数大则跳出循环，得到eid
        #     uncover_traj_num_e = len(set(lookup_table[lk_edges[i].id] + t)) - len(t)
        #     if uncover_traj_num_e >= len(lookup_table[lk_edges[i + 1].id]):
        #         target_e_idx = i
        #         break
        # eid = lk_edges[target_e_idx].id

        lk_edges = list(set(lookup_table.keys()).difference(set(em)))
        target_e_idx = 0
        for i in range(len(lk_edges) - 1):
            # 因为lookup table排好序，只要这个uncover大小比下一个边覆盖的轨迹数大则跳出循环，得到eid
            uncover_traj_num_e = len(set(lookup_table[lk_edges[i]] + t)) - len(t)
            if uncover_traj_num_e >= len(lookup_table[lk_edges[i + 1]]):
                target_e_idx = i
                break
        eid = lk_edges[target_e_idx]

        # 取差集 将选取到的边从边集去掉
        em.append(e[eid])
        edges = list(set(edges).difference(em))

        deta_s = 0
        route_pai = []
        t_pai = []

        # 第一次遍历 route为空，跳转if判断
        for r in route:

            # # # 给em添加基于路径的新增边
            # con_eg = []
            # for eg in r.e_list:
            #     con_eg += eg.left
            #     con_eg += eg.right
            #
            # for c_e in con_eg:
            #     if c_e in lookup_table.keys():
            #         em.append(e[c_e])
            # edges = list(set(edges).difference(set(em)))

            # 拷贝对象 直接赋值会传引用，r_pai改变r也会改变
            r_pai = Route(r.id, None, None, [])
            # r_pai.e_list = []
            cr = 0

            for re in r.e_list:
                cr += re.length
                r_pai.add_edge(re)

            t_pai_pai = t[:]
            cr_pai = cr
            deta_s_pai = 0
            em_pai = em[:]
            flag = True

            while flag:
                ec = []
                choose_e_id = -1
                new_traj_num = 0
                # 寻找连接路径的路段
                for re in em_pai:
                    if r_pai.start == re.end or r_pai.end == re.start:
                        # print('finding...')
                        if cr_pai + re.length < budget:
                            # 寻找有最多新加轨迹的路段
                            if re.id in lookup_table.keys():
                                if len(set(lookup_table[re.id] + t_pai_pai)) - len(t_pai_pai) > new_traj_num:
                                    choose_e_id = re.id
                                    new_traj_num = len(set(lookup_table[re.id] + t_pai_pai))
                                ec.append(re)
                if choose_e_id == -1:
                    flag = False
                    break

                # 添加路段进路径中
                # print('adding...')
                r_pai.add_edge(e[choose_e_id])

                # 更新已覆盖的轨迹集
                t_pai_pai = list(set(lookup_table[choose_e_id] + t_pai_pai))

                # 更新 em‘
                em_pai.append(e[choose_e_id])

                deta_s_pai = len(t_pai_pai) - len(t)

            if deta_s_pai > deta_s:
                deta_s = deta_s_pai
                # 将 r 替换成 r_pai
                route_pai = list(set(route).difference([r]))
                route_pai.append(r_pai)
                t_pai = t_pai_pai[:]

        # 判断是否不相等
        if compare_r(route, route_pai):
            # print('not equal route, route_pai len{}, route len:{}'.format(len(route_pai), len(route)))
            route = route_pai[:]
            t = t_pai[:]
            for r in route:
                # for re in r.e_list:
                em = list(set(em).difference(r.e_list))
        elif len(route) < k:
            # 如果是连通的，则添加进原有的路径中，否则新加路径
            ro_flag = False
            for ro in route:
                ro_flag = ro.add_edge(e[eid])
                if ro_flag:
                    break
            if not ro_flag:
                r = Route(r_idx, None, None, [])
                # r.e_list = []
                r.add_edge(e[eid])
                route.append(r)
                r_idx += 1

            t = list(set(lookup_table[eid] + t))

            # print('len t:{}'.format(len(t)))

            em = list(set(em).difference([e[eid]]))

        del lookup_table[eid]

    s = len(t) / len(trajs)

    for r in route:
        print('id:{}, length:{}'.format(r.id, len(r.e_list)))

    print('score:{}'.format(s))

    return route, s