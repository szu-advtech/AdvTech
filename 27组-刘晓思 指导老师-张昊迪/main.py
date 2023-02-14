from common.graph import read_graph
from common.utils import get_grid_dict, store_grid_dict, read_grid_dict, creat_lookup_table, store_lookup_table, read_lookup_table
from common.trajectory import read_trajectories
from algorithm.connect_first import connect_first
from algorithm.coverage_first import coverage_first
from algorithm.maximum_weight import maximum_weight

import folium
import os

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
    lk_path = 'data/lookup_table.txt'
    lookup_table = read_lookup_table(lk_path)

    ratio = 0.06

    # 4km x 0.01 = 40  0.06
    epsilon = border_length * ratio


    # number of representation routes
    k = 6

    # budget
    budget = 10000

    method = {
        'con': {
            'algorithm': connect_first,
            'score': 0,
            'routes': [],
            'save_file': 'connect_first.html'
        },
        'max': {
            'algorithm': maximum_weight,
            'score': 0,
            'routes': [],
            'save_file': 'maximum_weight.html'
        },
        'cov': {
            'algorithm': coverage_first,
            'score': 0,
            'routes': [],
            'save_file': 'coverage_first.html'
        }

    }

    # 算法选择
    # algorithm_method = method['con']
    # algorithm_method = method['max']
    algorithm_method = method['cov']

    # 算法
    # routes, score = connect_first(trajs, g.e, budget, k,lookup_table)
    # routes, score = coverage_first(trajs, g.e, budget, k,lookup_table)
    # routes, score = maximum_weight(trajs, g.e, budget, k,lookup_table)

    routes, score = algorithm_method['algorithm'](trajs, g.e, budget, k, lookup_table)
    algorithm_method['routes'], algorithm_method['score'] = routes, score

    draw_red = []
    draw_blue = []

    for t in trajs:
        d = []
        for p in t.p_list:
            d.append([p.lat, p.lng])
        draw_red.append(d)

    for ro in routes:
        d = []
        for ei in ro.e_list:
            d.append([ei.start.lat,ei.start.lng])
            d.append([ei.end.lat, ei.end.lng])
        draw_blue.append(d)

    center_lat = (mbr.min_lat+mbr.max_lat)/2
    center_lon = (mbr.min_lng+mbr.max_lng)/2

    m1 = folium.Map(location=[center_lat, center_lon],
                   zoom_start=14.8,
                   tiles='https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png',
                   attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'
                   )

    for d in draw_blue:
        folium.PolyLine(  # polyline方法为将坐标用线段形式连接起来
            d,  # 将坐标点连接起来
            weight=3,  # 线的大小为3
            color='blue',  # 线的颜色为橙色
            opacity=0.8  # 线的透明度
        ).add_to(m1)  # 将这条线添加到刚才的区域m内

    folium.Rectangle(
        bounds=((mbr.min_lat, mbr.min_lng), (mbr.max_lat, mbr.max_lng)),
        color='gray',
        fill=False
    ).add_to(m1)

    m1.save(os.path.join('result/', algorithm_method['save_file']))  # 将结果以HTML形式保存到指定路径








