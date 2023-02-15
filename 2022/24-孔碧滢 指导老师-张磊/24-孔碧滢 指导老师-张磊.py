from pulp import *
import numpy as np
import math
import logging
import random
from functools import cmp_to_key
import copy

logging.basicConfig(level=logging.INFO)



def solver():
    sfcs = ["1", "2"]# f      
    vnfs = ["1", "2"]# i 
    servers = ["1", "2" ,"3"]# v
    y_edge_lp = {}
    y_cloud_lp = {}
    timer = {}
    t1 = 10
    t2 = 50
    # Backup cost of sfcs f
    w = {
        "1": 12,
        "2": 12,
    }  

    #每个server上的total resource 
    total_resource = {
        "1": 16,# origin18 test 30
        "2": 18,
        "3": 16,
    }

    # resouces demand of VNFs i of sfcs f
    beta = [   # vnfs
        # 1 2
        [8, 4], # 1 sfcs
        [8, 4], # 2
    ]
    beta = makeDict([sfcs, vnfs], beta, 0)
    # Resource demand on v before deploying static backups server v 被用了的资源
    a_resource = {
        "1": 8,
        "2": 12,
        "3": 4,
    }
    v_resource = {
        "1": 8,
        "2": 4,
    }
    # Server holding the VNF i of   f
    o = [   #vnfs
        # 1    2
        ["1", "2"], # 1 sfcs
        ["2", "3"], # 2
    ]
    o = makeDict([sfcs, vnfs], o, None)
    vnfs_of_sfc = {
        "1":["1", "2"],
        "2":["1", "2"],
    }


    def lpSolver():
        # global y_cloud_lp
        # -----------Problem------------------
        prob = LpProblem("minimize cost", sense = LpMinimize)

        
        # -----------Variables----------------
        # Creates a list of tuples containing all the possible y_f_i_v
        y1 = pulp.LpVariable.dicts('y1', indices = (sfcs, vnfs , servers), lowBound = 0,upBound = 1, cat = LpContinuous)
        y2 = pulp.LpVariable.dicts('y2', indices = sfcs, lowBound = 0,upBound = 1, cat = LpContinuous)

        
        # -----------objective-----------------
        # minimize total backup cost
        prob += pulp.lpSum([w[f] * y2[f] for f in sfcs])

        
        # -----------constraints---------------

        # total demand of deployed vnfs backup on server v <= remained resource of server v 
        for v in servers:
            prob += pulp.lpSum([beta[f][i] * y1[f][i][v] for f in sfcs for i in vnfs]) <= (total_resource[v] - a_resource[v])

        # init y1            
        for f in sfcs:
            for i in vnfs:
                v = o[f][i]
                prob += y1[f][i][v] == 0

        ## set y2 
        # for f in sfcs:
        #     prob += y2[f] == 1 - pulp.lpSum([y1[f][i][v] for i in vnfs for v in servers])

        # constraint y1 and y2
        for f in sfcs:
            for i in vnfs:
                prob += 1-pulp.lpSum([y1[f][i][v] for v in servers]) <= y2[f]

        #-------------Solving--------------------
        prob.solve()
    
        #------------ static backup ------------

        for v in prob.variables():
            logging.info('{} = {}'.format(v.name, v.varValue))

        logging.info('backup cost = {}'.format(prob.objective))

        # 用词典存储结果key为三维tuple， value为int
        
        for f in sfcs:
            for i in vnfs:
                for v in servers:
                    y_edge_lp[(f, i, v)] = y1[f][i][v].value()
        
        for f in sfcs:
            y_cloud_lp[f] = y2[f].value()
        # y_cloud_lp = dict(sorted(y_cloud_lp.items(), reverse=True))

    lpSolver()
    y_edge = {}
    y_cloud = {}

    # init y_edge
    for f in sfcs:
        for i in vnfs:
            for v in servers:
                y_edge[(f,i,v)] = 0

    logging.info('y_cloud_lp: {}'.format(y_cloud_lp))
    logging.info('y_edge_lp: {}'.format( y_edge_lp))
    for f in sfcs:
        for i in vnfs:
            max = 0.0
            temp = ()
            for v in servers:
                if y_edge_lp[(f, i, v)] > max:
                    max = y_edge_lp[(f, i, v)]
                    temp = (f,i,v)
            y_edge[temp] = 1
            a_resource[temp[2]] += v_resource[i]
    logging.info('y_edge: {}'.format(y_edge))

    threshold = list(y_cloud_lp.values())[0]
    if threshold > 0:
        for key, value in y_cloud_lp.items():
            if value == threshold:
                y_cloud_lp[key] = 0
                y_cloud[key] = 1
                for i in vnfs:
                    for v in servers:
                        y_edge[(key, i, v)] = 0


    # -------------- dynamic backup ---------------------
    
    k_set= ["1","2"]                        # 随时到达的动态备份集
    kth = 1                                   # 记录目前的在备份 第k个 dynamic backup
    gama = {                                # resource demand of the dynamic backup k
        "1": 8,
        "2": 4,
    }        
    b_resource = {}                         # compute current resource demand on server v before deploy dynamic backup.
    # o_vnf = {"1":["1","2"], "2":["2","3"]}  # 原vnf1、2存放在哪些server上
    # o_staticBackup = {"1":[], "2":[]}       # 静态备份vnf 1、2 存放在哪些个server上
    o_vnf = {}
    o_staticBackup = {}
    placed_server = {"1":[], "2":[]}        # 放了 vnf1和静态备份的server合集
    deploying = {}
    deploying_list = []                  
    load_max = []
    n = 1
    bound_n = 10                           # n迭代的次数                        
    eta = {}
    overflow = True
    delta = {}
    x = {}
    p = 2
    epsilon = random.uniform(1,(1+p)/p)
    remaining_servers = copy.deepcopy(servers)
    qualified_server = ""   
    
    # init varable
    for k in k_set:
        for v in servers:
            x[(k, v)] = 0
      
    for f in sfcs:
        for i in vnfs:
            if i not in o_vnf:
                o_vnf[i] = []
            o_vnf[i].append(o[f][i])

    logging.info('o_vnf = {}'.format(o_vnf))
        
    while kth < len(k_set)+1 and n < bound_n:
        logging.info('n = {}'.format(n))
        for v in servers:
            eta[(n, v)] = 0  
        remaining_servers = copy.deepcopy(servers)
        placed_server = {"1":[], "2":[]}
        # 先检查有没有可恢复的云到边缘
        for t in sfcs:
            if t in timer:
                if timer[t] == t1:
                     # 调用static_backup看看能不能deploy边缘上去，如果y_cloud[f] = 0， 说明边缘有位置
                    lpSolver()
                    if y_cloud_lp[t] == 0:
                        #update y_edge and a_resource
                        y_cloud[t] = 0
                        for f in sfcs:
                            for i in vnfs:
                                max = 0.0
                                temp = ()
                                for v in servers:
                                    if y_edge_lp[(f, i, v)] > max:
                                        max = y_edge_lp[(f, i, v)]
                                        temp = (f,i,v)
                                y_edge[temp] = 1
                                a_resource[v] += v_resource[i]            #update resource
                        del timer[t]
                    else:
                        timer[t] = 0
                elif timer[t] == t2:
                    # 二话不说，直接把云上的布置回边缘上去。每一次放一个就选一个load最低的server放上去
                    y_cloud[t] = 0
                    for i in vnfs_of_sfc[t]:
                        for v in servers:
                            sb_used = 0
                            for f in sfcs:
                                for i in vnfs:
                                    if y_edge[(f,i,v)] == 1:
                                        sb_used += beta[f][i]
                                        o_staticBackup[i] = []
                                        o_staticBackup[i] += [v]
                            b_resource[v] = total_resource[v] - (a_resource[v] + sb_used)
                        logging.info('b_resource = {}'.format(b_resource))    
                            
                        for i in vnfs:
                            for num in o_vnf[i]:
                                placed_server[i].append(num)
                                
                            for num in o_staticBackup[i]:
                                placed_server[i].append(num)
                                
                        for v in servers:
                            deploying[v] = b_resource[v]/total_resource[v]
                        deploying_list = list(deploying.values())
                        load_max.append(sorted(deploying_list, key=float)[-1])
                        logging.info('load_max = {}'.format(load_max[0]))
                        y_edge[(t,i,load_max)] = 1
                        a_resource[load_max] += v_resource[i]  
                    timer.remove(t)
        overflow = True
        # 计算静态备份vnf放在哪些个server上
        # 计算server v 被用掉的资源
        for v in servers:
            for f in sfcs:
                for i in vnfs:
                    if i not in o_staticBackup:
                        o_staticBackup[i] = []
                    if y_edge[(f,i,v)] == 1:
                        o_staticBackup[i] += [v]
            b_resource[v] = total_resource[v] - (a_resource[v])
        logging.info('b_resource = {}'.format(b_resource))    
            
        for i in vnfs:
            for num in o_vnf[i]:
                placed_server[i].append(num)
                
            for num in o_staticBackup[i]:
                placed_server[i].append(num)
                
        for v in servers:
            deploying[v] = b_resource[v]/total_resource[v]
        deploying_list = list(deploying.values())
        load_max.append(sorted(deploying_list, key=float)[-1])
        logging.info('load_max = {}'.format(load_max[0]))

        # 计算可用的server
        
        for v in servers:
            sum = 0
            for j in range(kth):
                logging.info('j = {} k_set[j] = {} '.format(j, k_set[j]))
                sum += gama[k_set[j]] / total_resource[v] * x[(k_set[j],v)]
                logging.info('sum = {}'.format(sum))
            if (b_resource[v]/total_resource[v] + sum) > 1:
                remaining_servers.remove[v]
        
        # Backup adjustment if there is no qualified servers then terminate the algorithm and execute 
        if len(remaining_servers) == 0:
            overflow = False
            w_sorted = dict(sorted(w.items(), key=lambda x: x[1]))
            logging.info('w = {}'.format(w))
            logging.info('w_sorted = {}'.format(w_sorted))
            min_cost_sfc = list(w_sorted.keys())[0]
            y_cloud[min_cost_sfc] = 1           # 找到花费最小的server，把它布置到云上
            for v in servers:
                for i in vnfs:
                    # 怎么删除原来写死的 vnf、静态备份放在哪个server上。根据o知道的原来的vnf放在哪里，根据y_edge知道静态备份放在哪里
                    if y_edge[(min_cost_sfc, i, v)] == 1:
                        a_resource[v] -= beta[min_cost_sfc][i]
                        y_edge[(min_cost_sfc, i, v)] = 0
            for i in vnfs:
                a_resource[o[min_cost_sfc][i]] -= beta[min_cost_sfc][i]
            logging.info('o = {} min_cost_sfc = {}'.format(o, min_cost_sfc))        
            del o[min_cost_sfc]
            if min_cost_sfc in timer:
                timer[min_cost_sfc] = 0
                
        # 计算delta
        while overflow == True:
            overflow = False
            for v in remaining_servers:
                for k in k_set:
                    delta[(n,k,v)] = gama[k]/total_resource[v]*load_max[n-1]

            # server按照load规则，排序
            for num in placed_server[k_set[kth-1]]:
                if (num in remaining_servers): 
                    remaining_servers.remove(num)
            def custom_sort(a,b):
                if pow(epsilon, eta[(n,a)] + delta[(n,k_set[kth-1],a)]) - pow(epsilon, eta[(n,a)]) > pow(epsilon, eta[(n,b)] + delta[(n,k_set[kth-1],b)]) - pow(epsilon, eta[(n,b)]):
                    return -1
                elif pow(epsilon, eta[(n,a)] + delta[(n,k_set[kth-1],a)]) - pow(epsilon, eta[(n,a)]) < pow(epsilon, eta[(n,b)] + delta[(n,k_set[kth-1],b)]) - pow(epsilon, eta[(n,b)]):
                    return 1
                else:
                    return 0
            sorted(remaining_servers, key=cmp_to_key(custom_sort))
            logging.info('remaining_server = {}'.format(remaining_servers))
            
            if len(remaining_servers) != 0:
                qualified_server = remaining_servers[0]
                if eta[(n, qualified_server)] + delta[(n, k_set[kth-1], qualified_server)] > math.log(p/(p-1) * len(remaining_servers), epsilon):
                    load_max.append(load_max[-1] * 2)
                    for v in servers:
                        eta[(n, v)] = 0
                        overflow = True
                else:
                    x[(k_set[kth-1], qualified_server)] = 1
                    a_resource[qualified_server] += v_resource[k_set[kth-1]]
                    logging.info("x{}{} = 1".format(k_set[kth-1], qualified_server))
                    eta[(n, qualified_server)] = eta[(n, qualified_server)] + delta[(n, k_set[kth-1], qualified_server)]
                    kth += 1
                    n=1
                n+=1
            else:
                n+=1

            logging.info('x = {}'.format(x))
solver()
