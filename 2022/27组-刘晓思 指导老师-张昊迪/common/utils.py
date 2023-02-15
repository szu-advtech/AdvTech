import math
from common.point import Point
from tqdm import tqdm


DEGREES_TO_RADIANS = math.pi / 180
RADIANS_TO_DEGREES = 1 / DEGREES_TO_RADIANS
EARTH_MEAN_RADIUS_METER = 6371008.7714
DEG_TO_KM = DEGREES_TO_RADIANS * EARTH_MEAN_RADIUS_METER
LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05

INF = 10e9

class Segment:
    '''

    轨迹中的路段，传入路段的首尾两个轨迹点（Point类型）

    '''
    def __init__(self, start, end):
        self.start = start
        self.end = end

    # 计算路段长度
    def get_length(self):
        return distance(self.start, self.end)

# 计算a，b两点的距离，a，b类型为Point
def distance(a, b):
    """
    Calculate haversine distance between two GPS points in meters
    Args:
    -----
        a,b: SPoint class
    Returns:
    --------
        d: float. haversine distance in meter
    """
    if a == b:
        return 0.0
    delta_lat = math.radians(b.lat - a.lat)
    delta_lng = math.radians(b.lng - a.lng)
    h = math.sin(delta_lat / 2.0) * math.sin(delta_lat / 2.0) + math.cos(math.radians(a.lat)) * math.cos(
        math.radians(b.lat)) * math.sin(delta_lng / 2.0) * math.sin(delta_lng / 2.0)
    c = 2.0 * math.atan2(math.sqrt(h), math.sqrt(1 - h))
    d = EARTH_MEAN_RADIUS_METER * c
    return d


# 角度距离
def angle_dist(e, l):
    ll = l.get_length()
    angle_e = bearing(e.start, e.end) * DEGREES_TO_RADIANS
    angle_l = bearing(l.start, l.end) * DEGREES_TO_RADIANS
    deta = abs(angle_e - angle_l)
    if 2*math.pi - deta < deta:
        deta = 2*math.pi - deta
    if deta < math.pi/2:
        dist = ll * math.sin(deta)
    else:
        dist = ll
    return dist

# 边和线段的距离
def line_dist(e, l):

    # 计算l的首尾两点的投影点、移动率、和投影距离
    s_p, s_r, s_d = project_dist(l.start, e)
    e_p, e_r, e_d = project_dist(l.end, e)

    if s_r<0 or s_r>1 or e_r<0 or e_r>1:
        return INF

    # 计算垂直距离
    if (s_d + e_d) != 0:
        perpendicular_dist = (pow(s_d,2) + pow(e_d,2)) / (s_d + e_d)
    else:
        perpendicular_dist = 0

    # 计算平行距离
    parallel_dist = min(distance(e.start, s_p), distance(e.end, e_p))

    # print('perpendicular_dist:',perpendicular_dist)
    # print('parallel_dist:', parallel_dist)
    # print('angle_dist:', angle_dist(e, l))

    return perpendicular_dist + parallel_dist + angle_dist(e, l)



# http://www.movable-type.co.uk/scripts/latlong.html
# 计算方位角
def bearing(a, b):
    """
    Calculate the bearing of ab
    """
    pt_a_lat_rad = math.radians(a.lat)
    pt_a_lng_rad = math.radians(a.lng)
    pt_b_lat_rad = math.radians(b.lat)
    pt_b_lng_rad = math.radians(b.lng)
    y = math.sin(pt_b_lng_rad - pt_a_lng_rad) * math.cos(pt_b_lat_rad)
    x = math.cos(pt_a_lat_rad) * math.sin(pt_b_lat_rad) - math.sin(pt_a_lat_rad) * math.cos(pt_b_lat_rad) * math.cos(pt_b_lng_rad - pt_a_lng_rad)
    bearing_rad = math.atan2(y, x)
    return math.fmod(math.degrees(bearing_rad) + 360.0, 360.0)


def cal_loc_along_line(a, b, rate):
    """
    convert rate to gps location
    """
    lat = a.lat + rate * (b.lat - a.lat)
    lng = a.lng + rate * (b.lng - a.lng)
    return Point(lat, lng)

# 点到边的垂直距离
def project_dist(p, l):
    """
    Args:
    -----
    a,b: start/end GPS location of a road segment
    t: raw point
    Returns:
    -------
    project: projected GPS point on road segment
    rate: rate of projected point location to road segment
    dist: haversine_distance of raw and projected point
    """
    a = l.start
    b = l.end
    ab_angle = bearing(a, b)
    at_angle = bearing(a, p)
    ab_length = distance(a, b)
    at_length = distance(a, p)
    delta_angle = at_angle - ab_angle
    meters_along = at_length * math.cos(math.radians(delta_angle))
    if ab_length == 0.0:
        rate = 0.0
    else:
        rate = meters_along / ab_length
    if rate >= 1:
        projection = Point(b.lat, b.lng)
        rate = 1.0
    elif rate <= 0:
        projection = Point(a.lat, a.lng)
        rate = 0.0
    else:
        projection = cal_loc_along_line(a, b, rate)
    dist = distance(p, projection)
    return projection, rate, dist


# 创建网格与路网边的对应字典 key: grid id(x,y), value: eid
# 传入参数：创建的路网边界， 网格大小， 路网边集
def get_grid_dict(mbr, grid_size, edges):

    # 一个网格单元占的纬度经度
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    # 最大的网格单元
    max_xid = int((mbr.max_lat - mbr.min_lat) / lat_unit) + 1
    max_yid = int((mbr.max_lng - mbr.min_lng) / lng_unit) + 1

    # 网格-边字典集
    grid_dict = {}

    for e in edges:
        pre_lat = e.start.lat
        pre_lng = e.start.lng
        pre_locgrid_x = max(1, int((pre_lat - mbr.min_lat) / lat_unit) + 1)
        pre_locgrid_y = max(1, int((pre_lng - mbr.min_lng) / lng_unit) + 1)

        if (pre_locgrid_x, pre_locgrid_y) not in grid_dict.keys():
            grid_dict[(pre_locgrid_x, pre_locgrid_y)] = [e.id]
        else:
            grid_dict[(pre_locgrid_x, pre_locgrid_y)].append(e.id)

        lat = e.end.lat
        lng = e.end.lng
        locgrid_x = max(1, int((lat - mbr.min_lat) / lat_unit) + 1)
        locgrid_y = max(1, int((lng - mbr.min_lng) / lng_unit) + 1)

        if (locgrid_x, locgrid_y) not in grid_dict.keys():
            grid_dict[(locgrid_x, locgrid_y)] = [e.id]
        else:
            grid_dict[(locgrid_x, locgrid_y)].append(e.id)

        mid_x_num = abs(locgrid_x - pre_locgrid_x)
        mid_y_num = abs(locgrid_y - pre_locgrid_y)

        # 在端点两边的情况
        if mid_x_num > 1 and mid_y_num <= 1:
            for mid_x in range(1, mid_x_num):
                if (min(pre_locgrid_x, locgrid_x) + mid_x, locgrid_y) not in grid_dict.keys():
                    grid_dict[(min(pre_locgrid_x, locgrid_x) + mid_x, locgrid_y)] = [e.id]
                else:
                    grid_dict[(min(pre_locgrid_x, locgrid_x) + mid_x, locgrid_y)].append(e.id)

        elif mid_x_num <= 1 and mid_y_num > 1:
            for mid_y in range(1, mid_y_num):
                if (locgrid_x, min(pre_locgrid_y, locgrid_y) + mid_y) not in grid_dict.keys():
                    grid_dict[(locgrid_x, min(pre_locgrid_y, locgrid_y) + mid_y)] = [e.id]
                else:
                    grid_dict[(locgrid_x, min(pre_locgrid_y, locgrid_y) + mid_y)].append(e.id)

        # 中间情况
        elif mid_x_num > 1 and mid_y_num > 1:
            ttl_num = mid_x_num + mid_y_num + 1
            for mid in range(1, ttl_num):
                # 中间坐标
                mid_xid = min(lat, pre_lat) + mid * abs(lat - pre_lat) / ttl_num
                mid_yid = min(lng, pre_lng) + mid * abs(lng - pre_lng) / ttl_num

                mid_locgrid_x = max(1, int((mid_xid - mbr.min_lat) / lat_unit) + 1)
                mid_locgrid_y = max(1, int((mid_yid - mbr.min_lng) / lng_unit) + 1)

                if (mid_locgrid_x, mid_locgrid_y) not in grid_dict.keys():
                    grid_dict[(mid_locgrid_x, mid_locgrid_y)] = [e.id]
                else:
                    grid_dict[(mid_locgrid_x, mid_locgrid_y)].append(e.id)


    for k, v in grid_dict.items():
        grid_dict[k] = list(set(v))

    return grid_dict, max_xid, max_yid


# 存储网格字典 (x,y):[e.id...]
def store_grid_dict(grid_dict, path):
    with open(path, 'w') as f:
        for k, v in grid_dict.items():
            f.write('{},{}'.format(k[0], k[1]))
            for i in v:
                f.write(',{}'.format(i))
            f.write('\n')


# 读取文件中的网格字典
def read_grid_dict(path):
    grid_dict = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            attrs = line.rstrip().split(',')
            grid_dict[(int(attrs[0]), int(attrs[1]))] = [int(attrs[2])]

            if len(attrs) > 3:
                for i in attrs[3:]:
                    grid_dict[(int(attrs[0]), int(attrs[1]))].append(int(i))

    return grid_dict


# 创建lookup table
# 传入参数：网格字典，网格大小，网格边界，路网边集，轨迹
def creat_lookup_table(grid_dict, grid_size, mbr, trajs, edges, epsilon):

    # 一个网格单元占的纬度经度
    lat_unit = LAT_PER_METER * grid_size
    lng_unit = LNG_PER_METER * grid_size

    lookup_table = {}

    print('creating lookup table...')

    # 遍历轨迹
    for t in tqdm(trajs):
        p_list = t.p_list
        # 对轨迹序列进行每个边的构建segment，然后寻找能cover这条路段的路网边

        # segment 的起始点
        start = p_list[0]
        # 计算网格所在位置
        s_locgrid_x = max(1, int((start.lat - mbr.min_lat) / lat_unit) + 1)
        s_locgrid_y = max(1, int((start.lng - mbr.min_lng) / lng_unit) + 1)
        # 计算网格对应的边（上下左右9个格）
        # 起始点附近的边集，需要统计同时出现在start和end的附近的路网边
        s_e_list = []
        if (s_locgrid_x, s_locgrid_y) in grid_dict.keys():
            s_e_list += grid_dict[(s_locgrid_x, s_locgrid_y)]
        if (s_locgrid_x, s_locgrid_y-1) in grid_dict.keys():
            s_e_list += grid_dict[(s_locgrid_x, s_locgrid_y-1)]
        if (s_locgrid_x, s_locgrid_y+1) in grid_dict.keys():
            s_e_list += grid_dict[(s_locgrid_x, s_locgrid_y+1)]
        if (s_locgrid_x-1, s_locgrid_y) in grid_dict.keys():
            s_e_list += grid_dict[(s_locgrid_x-1, s_locgrid_y)]
        if (s_locgrid_x-1, s_locgrid_y-1) in grid_dict.keys():
            s_e_list += grid_dict[(s_locgrid_x-1, s_locgrid_y-1)]
        if (s_locgrid_x-1, s_locgrid_y+1) in grid_dict.keys():
            s_e_list += grid_dict[(s_locgrid_x-1, s_locgrid_y+1)]
        if (s_locgrid_x+1, s_locgrid_y) in grid_dict.keys():
            s_e_list += grid_dict[(s_locgrid_x+1, s_locgrid_y)]
        if (s_locgrid_x+1, s_locgrid_y-1) in grid_dict.keys():
            s_e_list += grid_dict[(s_locgrid_x+1, s_locgrid_y-1)]
        if (s_locgrid_x+1, s_locgrid_y+1) in grid_dict.keys():
            s_e_list += grid_dict[(s_locgrid_x+1, s_locgrid_y+1)]

        for p in p_list[1:]:

            # 计算网格所在位置
            e_locgrid_x = max(1, int((p.lat - mbr.min_lat) / lat_unit) + 1)
            e_locgrid_y = max(1, int((p.lng - mbr.min_lng) / lng_unit) + 1)

            # 计算网格对应的边（上下左右9个格）
            # 起始点附近的边集，需要统计同时出现在start和end的附近的路网边
            e_e_list = []
            if (e_locgrid_x, e_locgrid_y) in grid_dict.keys():
                e_e_list += grid_dict[(e_locgrid_x, e_locgrid_y)]
            if (e_locgrid_x, e_locgrid_y-1) in grid_dict.keys():
                e_e_list += grid_dict[(e_locgrid_x, e_locgrid_y - 1)]
            if (e_locgrid_x, e_locgrid_y+1) in grid_dict.keys():
                e_e_list += grid_dict[(e_locgrid_x, e_locgrid_y + 1)]
            if (e_locgrid_x-1, e_locgrid_y) in grid_dict.keys():
                e_e_list += grid_dict[(e_locgrid_x - 1, e_locgrid_y)]
            if (e_locgrid_x-1, e_locgrid_y-1) in grid_dict.keys():
                e_e_list += grid_dict[(e_locgrid_x - 1, e_locgrid_y - 1)]
            if (e_locgrid_x-1, e_locgrid_y+1) in grid_dict.keys():
                e_e_list += grid_dict[(e_locgrid_x - 1, e_locgrid_y + 1)]
            if (e_locgrid_x+1, e_locgrid_y) in grid_dict.keys():
                e_e_list += grid_dict[(e_locgrid_x + 1, e_locgrid_y)]
            if (e_locgrid_x+1, e_locgrid_y-1) in grid_dict.keys():
                e_e_list += grid_dict[(e_locgrid_x + 1, e_locgrid_y - 1)]
            if (e_locgrid_x+1, e_locgrid_y+1) in grid_dict.keys():
                e_e_list += grid_dict[(e_locgrid_x + 1, e_locgrid_y + 1)]

            # 去除重复项
            e_e_list = list(set(e_e_list))

            near_edges = []

            # 寻找两端的点邻近的边集
            for i in e_e_list:
                if i in set(s_e_list):
                    near_edges.append(i)

            # 跟据范围选取到的边集计算cover，构建lookup table
            l = Segment(start, p)
            for eid in near_edges:
                if line_dist(edges[eid], l) <= epsilon:
                    if eid not in lookup_table.keys():
                        lookup_table[eid] = [t.id]
                    else:
                        lookup_table[eid].append(t.id)

            # 更新两端其中的起始点
            start = p
            s_e_list = e_e_list

    # 去除重复项
    for k, v in lookup_table.items():
        lookup_table[k] = list(set(v))

    lookup_table = dict(sorted(lookup_table.items(), key=lambda x: len(x[1]), reverse=True))

    return lookup_table


# 存储lookup table
def store_lookup_table(lookup_table, path):
    # 排序后存储，跟据列表长度来排序
    s_lookup_table = sorted(lookup_table.items(), key=lambda x: len(x[1]), reverse=True)
    with open(path, 'w') as f:
        for k, v in s_lookup_table:
            f.write(str(k))
            for i in v:
                f.write(',{}'.format(i))
            f.write('\n')

# 读取文件中的lookup table
def read_lookup_table(path):
    lookup_table = {}

    with open(path, 'r') as f:
        for line in f.readlines():
            attrs = line.rstrip().split(',')
            lookup_table[int(attrs[0])] = [int(attrs[1])]

            if len(attrs) > 2:
                for i in attrs[2:]:
                    lookup_table[int(attrs[0])].append(int(i))

    return lookup_table









