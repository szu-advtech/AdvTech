import numpy as np
import math


'''
将经纬度坐标转换为地图投影坐标。
地图投影坐标是将地球投影到平面上时产生的坐标系，这些坐标可以用来在平面地图上绘制点、线和面。
'''
def latlon2map(lon, lat):
    semimajoraxis = 6378137.0   #半长轴
    #将经度和纬度乘上一个系数来转换为弧度制。
    east = lon * 0.017453292519943295 #经度
    north = lat * 0.017453292519943295  #纬度
    t = math.sin(north)
    return semimajoraxis * east, 3189068.5 * math.log((1 + t) / (1 - t))

'''
将轨迹点转化到地图投影坐标，使用到lonlat2meters
输入：轨迹点列表
输出：列表中每一个点的地图投影坐标的列表
'''
def points2meter(points):
    rtn = []
    for p in points:
        lon_meter, lat_meter = latlon2map(lon=p[1], lat=p[0])
        rtn.append([lat_meter,lon_meter,p[2]])
    return rtn
'''
读取文件，把信息转化为轨迹三元组[经度，纬度，时间戳]
输入：原始文件
输出：轨迹文件
'''
def to_traj(file):
    traj = []
    f = open(file)
    for line in f:
        temp = line.strip().split(' ')
        if len(temp) < 3:
            continue
        traj.append([float(temp[0]), float(temp[1]), int(float(temp[2]))])
    f.close()
    return traj

if __name__ == '__main__':
    print('测试，无语法错误!')