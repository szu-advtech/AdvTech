from common.graph import read_graph
from common.utils import get_grid_dict, store_grid_dict, read_grid_dict, creat_lookup_table, store_lookup_table, read_lookup_table
from common.trajectory import read_trajectories
from algorithm.connect_first import connect_first
from algorithm.coverage_first import coverage_first
from algorithm.maximum_weight import maximum_weight

import matplotlib.pyplot as plt
import time

if __name__ == '__main__':
    # 路网图
    g_path = 'data/'
    g, mbr = read_graph(g_path)

    # 网格化的网格单元大小
    grid_size = 40

    # 利用划分网格进行查询
    grid_dict, max_xid, max_yid = get_grid_dict(mbr, grid_size, g.e)
    grid_dict_path = 'data/grid_dict.txt'
    store_grid_dict(grid_dict, grid_dict_path)

    # 查询边界长度
    border_length = 4000

    # 轨迹
    t_path = 'data/cd_test_trajs.txt'
    trajs, t_mbr = read_trajectories(t_path)

    # 查找表
    # lk_path = 'data/lookup_table.txt'
    # lookup_table = read_lookup_table(lk_path)


    # 设置参数列表，可做修改
    e_ratio_list = [0.06, 0.08, 0.1, 0.12, 0.14]
    b_ratio_list = [0.125, 0.25, 0.5, 1, 2]
    k_list = [1, 6, 11, 16, 21]

    # 默认：er:0.1 br:1, k:11 ?

    method = {
        'con':{
            'algorithm': connect_first,
            'score': [],
            'routes': [],
            'time': [],
        },
        'max': {
            'algorithm': maximum_weight,
            'score': [],
            'routes': [],
            'time': []
        },
        'cov':{
            'algorithm': coverage_first,
            'score': [],
            'routes': [],
            'time': []
        }

    }

    # e_ratio 的更改实验
    k = 6
    b_ratio = 1
    budget = border_length * b_ratio
    for e_ratio in e_ratio_list:
        epsilon = border_length * e_ratio
        lookup_table = creat_lookup_table(grid_dict, grid_size, mbr, trajs, g.e, epsilon)
        for key,v in method.items():
            start = time.time()
            routes, score = method[key]['algorithm'](trajs, g.e, budget, k, lookup_table)
            end = time.time()
            method[key]['time'].append(end - start)
            method[key]['score'].append(score)
            method[key]['routes'].append(routes)

    plt.figure(figsize=(5, 5))
    plt.margins(0.1)
    plt.xlabel('e_ratio')
    plt.ylabel('Score')
    plt.ylim([0.00, 1.00])
    plt.xticks(range(5), e_ratio_list)
    plt.plot(method['con']['score'], color='#FFCC33', marker='^', label='con')
    plt.plot(method['cov']['score'], color='#336699',marker='*', label='cov')
    plt.plot(method['max']['score'], color='#FF6666', marker='o', label='max')
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.margins(0.1)
    plt.xlabel('e_ratio')
    plt.ylabel('Runtime')
    plt.xticks(range(5), e_ratio_list)
    plt.plot(method['con']['time'], color='#FFCC33', marker='^', label='con')
    plt.plot(method['cov']['time'], color='#336699', marker='*', label='cov')
    plt.plot(method['max']['time'], color='#FF6666', marker='o', label='max')
    plt.legend(loc='lower right')
    plt.show()

    lk_path = 'data/lookup_table.txt'
    lookup_table = read_lookup_table(lk_path)

    method = {
        'con': {
            'algorithm': connect_first,
            'score': [],
            'routes': [],
            'time': [],
        },
        'max': {
            'algorithm': maximum_weight,
            'score': [],
            'routes': [],
            'time': []
        },
        'cov': {
            'algorithm': coverage_first,
            'score': [],
            'routes': [],
            'time': []
        }

    }

    b_ratio = 1
    budget = border_length * b_ratio
    for k in k_list:
        for key, v in method.items():
            start = time.time()
            routes, score = method[key]['algorithm'](trajs, g.e, budget, k, lookup_table)
            end = time.time()
            method[key]['time'].append(end - start)
            method[key]['score'].append(score)
            method[key]['routes'].append(routes)

    plt.figure(figsize=(5, 5))
    plt.margins(0.1)
    plt.xlabel('k')
    plt.ylabel('Score')
    plt.ylim([0.00, 1.00])
    plt.xticks(range(5), k_list)
    plt.plot(method['con']['score'], color='#FFCC33', marker='^', label='con')
    plt.plot(method['cov']['score'], color='#336699', marker='*', label='cov')
    plt.plot(method['max']['score'], color='#FF6666', marker='o', label='max')
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.margins(0.1)
    plt.xlabel('k')
    plt.ylabel('Runtime')
    plt.xticks(range(5), k_list)
    plt.plot(method['con']['time'], color='#FFCC33', marker='^', label='con')
    plt.plot(method['cov']['time'], color='#336699', marker='*', label='cov')
    plt.plot(method['max']['time'], color='#FF6666', marker='o', label='max')
    plt.legend(loc='lower right')
    plt.show()

    method = {
        'con': {
            'algorithm': connect_first,
            'score': [],
            'routes': [],
            'time': [],
        },
        'max': {
            'algorithm': maximum_weight,
            'score': [],
            'routes': [],
            'time': []
        },
        'cov': {
            'algorithm': coverage_first,
            'score': [],
            'routes': [],
            'time': []
        }

    }
    k = 6
    for b_ratio in b_ratio_list:
        budget = border_length * b_ratio
        for key, v in method.items():
            start = time.time()
            routes, score = method[key]['algorithm'](trajs, g.e, budget, k, lookup_table)
            end = time.time()
            method[key]['time'].append(end - start)
            method[key]['score'].append(score)
            method[key]['routes'].append(routes)

    plt.figure(figsize=(5, 5))
    plt.margins(0.1)
    plt.xlabel('b_ratio')
    plt.ylabel('Score')
    plt.ylim([0.00, 1.00])
    plt.xticks(range(5), b_ratio_list)
    plt.plot(method['con']['score'], color='#FFCC33', marker='^', label='con')
    plt.plot(method['cov']['score'], color='#336699', marker='*', label='cov')
    plt.plot(method['max']['score'], color='#FF6666', marker='o', label='max')
    plt.legend(loc='lower right')
    plt.show()

    plt.figure(figsize=(5, 5))
    plt.margins(0.1)
    plt.xlabel('b_ratio')
    plt.ylabel('Runtime')
    plt.xticks(range(5), b_ratio_list)
    plt.plot(method['con']['time'], color='#FFCC33', marker='^', label='con')
    plt.plot(method['cov']['time'], color='#336699', marker='*', label='cov')
    plt.plot(method['max']['time'], color='#FF6666', marker='o', label='max')
    plt.legend(loc='lower right')
    plt.show()

    # plt.title('')
    # plt.figure(figsize=(5, 5))
    # plt.margins(0.1)
    # # plt.xlabel('k')
    # # plt.xlabel('b_ratio')
    # plt.xlabel('e_ratio')
    # plt.ylabel('score')
    # plt.ylim([0.00, 1.00])
    # # plt.xticks(range(5), k_list)
    # # plt.xticks(range(5), b_ratio_list)
    # plt.xticks(range(5), e_ratio_list)
    # plt.plot(method['con']['score'], color='#FFCC33', marker='^', label='con')
    # plt.plot(method['cov']['score'], color='#336699',marker='*', label='cov')
    # plt.plot(method['max']['score'], color='#FF6666', marker='o', label='max')
    # plt.legend(loc='lower right')
    # plt.show()


    # # e_ratio = 0.06
    #
    # # 4km x 0.01 = 40  0.06
    # epsilon = border_length * e_ratio
    #
    # # number of representation routes
    # k = 6
    #
    # # budget
    # budget = 10000

    # 连接优先算法
    # routes, score = connect_first(trajs, g.e, budget, k,lookup_table)
    # routes, score = coverage_first(trajs, g.e, budget, k, lookup_table)
    # routes, score = maximum_weight(trajs, g.e, budget, k,lookup_table)

    # k = 6
    # b_ratio = 1
    # budget = border_length * b_ratio
    # lk_path = 'data/lookup_table.txt'
    # lookup_table = read_lookup_table(lk_path)
    #
    # time_list = []
    # index_list = []
    # for key, v in method.items():
    #     start = time.time()
    #     routes, score = method[key]['algorithm'](trajs, g.e, budget, k, lookup_table)
    #     end = time.time()
    #     time_list.append(end - start)
    #     index_list.append(key)
    #     method[key]['score'].append(score)
    #     method[key]['routes'].append(routes)
    #
    # # plt.figure(figsize=(5, 5))
    # plt.margins(0.1)
    # plt.xlabel('method')
    # plt.ylabel('running time')
    # plt.xticks(range(3), index_list)
    # plt.bar(index_list, height=time_list, color='#336699')
    # plt.show()