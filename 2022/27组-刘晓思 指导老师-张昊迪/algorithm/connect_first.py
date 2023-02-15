from common.route import Route

def connect_first(trajs, e, budget, k, lookup_table):
    # connect-first algorithm
    routes = []

    # covered trajectories
    t = []

    r_idx = 0

    while len(routes) < k:
        # 选取the first edge for a route
        edges = list(lookup_table.keys())
        target_e_idx = 0
        for i in range(len(edges) - 1):
            # 因为lookup table排好序，只要这个uncover大小比下一个边覆盖的轨迹数大则跳出循环，得到eid
            uncover_traj_num_e = len(set(lookup_table[edges[i]] + t)) - len(t)
            if uncover_traj_num_e >= len(lookup_table[edges[i + 1]]):
                target_e_idx = i
                break
        eid = edges[target_e_idx]

        # 路径
        r = Route(r_idx, None, None, [])
        # 添加第一条路径边
        r.add_edge(e[eid])

        # 添加这条路径边覆盖的轨迹
        t = list(set(lookup_table[eid] + t))

        # 计算路径的budget 边的budget用边的长度
        cr = e[eid].length

        # print('route id:{}'.format(r_idx))
        # for i in r.e_list:
        #   print(i)

        # 添加更多路径边，查找基于第一条边的连接边
        while cr < budget:
            # 将首尾的连接边加入计算(得到的是id)
            ec = list(set(r.e_list[0].left + r.e_list[-1].right))
            # print(ec)

            deta_s = 0
            # r_pai = r
            t_pai = []
            cr_pai = 0
            choose_e_idx = -1

            for e_idx in ec:
                deta_s_pai = 0
                ce = e[e_idx].length
                if cr + ce <= budget:
                    if e_idx in lookup_table.keys():
                        deta_s_pai = len(set(lookup_table[e_idx] + t)) - len(t)
                # 如果没有新加入的边会陷入死循环。。
                if deta_s_pai == 0:
                    continue
                if deta_s_pai > deta_s and choose_e_idx not in r.e_list:
                    # r_pai = r
                    # r_pai.add_edge(e[e_idx])
                    choose_e_idx = e_idx
                    deta_s = deta_s_pai
                    t_pai = list(set(lookup_table[e_idx] + t))
                    cr_pai = cr + ce

            # 判断两条路径是否一样，并且不为第一个路径
            # if set(r.e_list) == set(r_pai.e_list):
            #     break
            # else:
            #     r = r_pai
            #     cr = cr_pai
            #     t = t_pai

            # 如果找不到更多的边，并且budget特别小，则添加包含多轨迹的路段
            if choose_e_idx == -1:
                max_len_idx = -1
                max_len = 0
                for e_idx in ec:
                    # 添加
                    if e_idx in lookup_table.values() and len(lookup_table[e_idx]) > max_len:
                        max_len_idx = e_idx
                        ce = e[e_idx].length
                if max_len_idx != -1:
                    r.add_edge(e[max_len_idx])
                    cr += ce
                    # print('continue')
                if max_len_idx == -1:
                    # print('break')
                    break

            r.add_edge(e[choose_e_idx])
            cr = cr_pai
            t = t_pai

                # print('route id:{}'.format(r_idx))
                # for i in r.e_list:
                #   print(i)

        # 添加路径
        routes.append(r)

        # 添加边

        print('route id:{} , length:{}'.format(r_idx, len(r.e_list)))
        # for i in r.e_list:
        #    print(i)

        r_idx += 1

    # representation score
    score = len(t) / len(trajs)
    print(score)

    return routes, score
