import pandas as pd
import os
from common.point import Point

'''


path = 'chengdu_trajectory_10000/'
files = os.listdir(path)

max_lat = -1000
max_lon = -1000
min_lat = 1000
min_lon = 1000

for file in files:
    t = pd.read_csv(path+file)
    max_lat = max(max_lat, max(t['lat']))
    min_lat = min(min_lat, min(t['lat']))
    max_lon = max(max_lon, max(t['lon']))
    min_lon = min(min_lon, min(t['lon']))

info = 'max_lat:{}\nmax_lon:{}\nmin_lat:{}\nmin_lon:{}'.format(max_lat, max_lon, min_lat, min_lon)
print(info)

with open('traj_mbr.txt', 'w') as f:
    f.write(info)
    
'''

LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05

lat_3km = LAT_PER_METER * 3000
lon_3km = LNG_PER_METER * 3000

print('traj lat_3km：{}， lon_3km:{}'.format(lat_3km, lon_3km))

mbr = []
with open('traj_mbr.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        attr = line.split(':')
        mbr.append(float(attr[1]))

# print(mbr)

lat_length = (mbr[0]-mbr[2]) / LAT_PER_METER
lon_length = (mbr[1]-mbr[3]) / LNG_PER_METER

print('traj lat_length：{}， lon_length：{}'.format(lat_length, lon_length))

# 轨迹区域
# traj lat_length：8313.247932398686， lon_length：7464.348731403841

# 分隔轨迹区域
# 以3km划分轨迹范围，分成9块

# id, mbr, traj
traj_region = {}

lat_number = int(lat_length // 3000 + 1)
lon_number = int(lon_length // 3000 + 1)

print('lat_number:{}, lon_number:{}'.format(lat_number, lon_number))

border_min_lat = mbr[2]
for i in range(lat_number):
    border_min_lon = mbr[3]
    for j in range(lon_number):
        region = {}
        region['min_lat'] = border_min_lat
        region['max_lat'] = border_min_lat + lat_3km
        region['min_lon'] = border_min_lon
        region['max_lon'] = border_min_lon + lon_3km
        traj_region[i*3+j] = region
        border_min_lon += lon_3km
    border_min_lat += lat_3km

print(traj_region)




def p_in_region(p, region):
    if p.lat <= region['max_lat'] and p.lat >= region['min_lat'] and p.lng <= region['max_lon'] and p.lng >= region['min_lon']:
        return True
    else:
        return False

# 根据区域划分轨迹
path = 'chengdu_trajectory_10000/'
files = os.listdir(path)

traj_num = 0

trajs = {}

r = traj_region[0]

for file in files:
    t = pd.read_csv(path+file)
    lat_list = t['lat']
    lon_list = t['lon']
    traj = []
    for i in range(len(lat_list)):
        p = Point(lat_list[i], lon_list[i])
        if p_in_region(p, r):
            traj.append(p)
        else:
            # 断开的点，存储轨迹
            if len(traj) > 0:
                # 轨迹长度大于10
                if len(traj) > 10:
                    trajs[traj_num] = traj
                    traj_num += 1
                traj = [p]

print(traj_num)

with open('cd_test_trajs.txt', 'w') as f:
    for tr in trajs.values():
        for p in tr:
            f.write('{} {} '.format(p.lat, p.lng))
        f.write('\n')






