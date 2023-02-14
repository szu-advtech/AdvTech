import numpy as np
import json
from common.utils import distance
from common.point import Point

segment = np.load('segment_all_dict.npy',allow_pickle=True)

# for s in segment:
#     print(s)


# with open('segment.txt', 'w') as f:
#     f.write(str(segment))


d = dict(enumerate(segment.flatten(), 1))[1]

# 路段数量 20063
print(len(d))

print(d['1'])

# {'link_id': '1', 'osm_way_id': '25373389', 'name': '人民中路一段', 'length': 47.21, 'node_a_id': '441187183', 'node_a': (30.6588526, 104.0647365), 'node_b_id': '0', 'node_b': (30.6592004, 104.0650195)}

# max_lat = -1000
# max_lon = -1000
# min_lat = 1000
# min_lon = 1000

LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05

deta_lat = LAT_PER_METER * 500
deta_lon = LNG_PER_METER * 500

min_lat = 30.65541009760488 - deta_lat
max_lat = 30.68238970863773 + deta_lat
min_lon = 104.03969948534711 - deta_lon
max_lon = 104.07480006725845 + deta_lon

region_edges = []
region_node = {}

for edge in d.values():

    lat = edge['node_a'][0]
    lon = edge['node_a'][1]

    # 筛选符合范围的边和点
    if lat <= max_lat and lon <= max_lon and lat >= min_lat and lon >= min_lon:
        region_edges.append([edge['node_a_id'], edge['node_b_id'], edge['length']])
        region_node[edge['node_a_id']] = edge['node_a']
        region_node[edge['node_b_id']] = edge['node_b']
        continue

    lat = edge['node_b'][0]
    lon = edge['node_b'][1]

    if lat <= max_lat and lon <= max_lon and lat >= min_lat and lon >= min_lon:
        region_edges.append([edge['node_a_id'], edge['node_b_id'], edge['length']])
        region_node[edge['node_a_id']] = edge['node_a']
        region_node[edge['node_b_id']] = edge['node_b']

print('edges:{}'.format(len(region_edges)))
print('node:{}'.format(len(region_node)))

with open('cd_test_edge.txt', 'w') as f:
    f.write('{}\n'.format(len(region_edges)))
    for e in region_edges:
        print(e)
        print(len(e))
        f.write('{} {} {}\n'.format(e[0], e[1], e[2]))

with open('cd_test_node.txt', 'w') as f:
    f.write('{}\n'.format(len(region_node)))
    for id,p  in region_node.items():
        f.write('{} {} {}\n'.format(id, p[0], p[1]))


# info = 'max_lat:{}\nmax_lon:{}\nmin_lat:{}\nmin_lon:{}'.format(max_lat, max_lon, min_lat, min_lon)
# print(info)
#
# with open('region_mbr.txt', 'w') as f:
#     f.write(info)