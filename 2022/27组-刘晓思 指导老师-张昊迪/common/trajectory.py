from .point import Point
from common.graph import Mbr

class Trajectory:
    '''
    轨迹：轨迹id，轨迹序列
    '''
    def __init__(self, id, p_list):
        self.id = id
        self.p_list = p_list

    def __str__(self):
        return 'tid:{}, list:{}'.format(self.id, [str(p) for p in self.p_list])


# 读取轨迹数据 传入：文件路径
def read_trajectories(file_path):
    t = []

    min_lat = 1000.0
    min_lng = 1000.0
    max_lat = -1000.0
    max_lng = -1000.0

    with open(file_path, 'r') as f:
        index = 0
        for line in f.readlines():
            attrs = line.rstrip().split(' ')
            i = 0
            p_list = []
            while i < len(attrs):
                lat = float(attrs[i])
                i += 1
                lng = float(attrs[i])
                i += 1
                # if lat<g_mbr.min_lat or lat>g_mbr.max_lat or lng<g_mbr.min_lng or lng>g_mbr.max_lng:
                #     continue
                p = Point(lat, lng)
                p_list.append(p)

                if lat < min_lat:
                    min_lat = lat
                if lng < min_lng:
                    min_lng = lng
                if lat > max_lat:
                    max_lat = lat
                if lng > max_lng:
                    max_lng = lng
            if len(p_list)<=0:
                continue
            tr = Trajectory(index, p_list)
            t.append(tr)
            index += 1
    print('number of trajectories:', index)

    mbr = Mbr(min_lat, min_lng, max_lat, max_lng)
    print(mbr)

    return t, mbr
