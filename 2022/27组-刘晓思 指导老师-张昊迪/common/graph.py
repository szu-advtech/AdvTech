from tqdm import tqdm

class Vertex:
    '''

    路网图中的点
    交点的id，以及经纬度坐标

    '''
    def __init__(self, id, lat, lng):
        self.id = id
        self.lat = lat
        self.lng = lng

    # 重写 == 方法，可以判断顶点是否相等
    def __eq__(self, other):

        return self.lat == other.lat and self.lng == other.lng

class Edge:
    '''

    start & end:路网边的两端的点，Vertex类型
    length:长度

    '''
    def __init__(self, id, start, end, length):
        self.id = id
        self.start = start
        self.end = end
        self.length = length
        # 左连接和右连接的邻居边的id
        self.left = []
        self.right = []

    def __str__(self):
        return 'eid_{}'.format(self.id)

class Graph:

    def __init__(self, v, e, v_num, e_num):
        self.v = v
        self.e = e
        self.v_num = v_num
        self.e_num = e_num

# 边界类，一个区域的最大最小边界
class Mbr:
    def __init__(self, min_lat, min_lng, max_lat, max_lng):
        self.min_lat = min_lat
        self.min_lng = min_lng
        self.max_lat = max_lat
        self.max_lng = max_lng

    def __str__(self):
        return 'max_lat:{}\nmax_lng:{}\nmin_lat:{}\nmin_lng:{}'.format(self.max_lat, self.max_lng, self.min_lat, self.min_lng)

# 读取路网图 传入：点和边文件的文件夹路径
def read_graph(path):
    v = {}

    min_lat = 1000.0
    min_lng = 1000.0
    max_lat = -1000.0
    max_lng = -1000.0

    with open(path+'cd_test_node.txt', 'r') as f:
        v_num = int(f.readline())

        index = 0

        for line in f.readlines():
            attrs = line.rstrip().split(' ')
            lat = float(attrs[1])
            lng = float(attrs[2])

            if lat < min_lat:
                min_lat = lat
            if lng < min_lng:
                min_lng = lng
            if lat > max_lat:
                max_lat = lat
            if lng > max_lng:
                max_lng = lng

            ve = Vertex(attrs[0], lat, lng)
            v[attrs[0]] = ve

            index += 1

    print('number of vertices:', len(v))

    # 稍微扩大一点点mbr边缘
    mbr = Mbr( min_lat-0.001, min_lng-0.001, max_lat+0.001, max_lng+0.001)
    # print('min_lat:{}, min_lng:{}'.format(min_lat, min_lng))

    e = []
    with open(path+'cd_test_edge.txt', 'r') as f:
        e_num = int(f.readline())

        index = 0

        for line in f.readlines():
            attrs = line.rstrip().split(' ')
            s_index = attrs[0]
            e_index = attrs[1]
            length = float(attrs[2])

            ed = Edge(index, v[s_index], v[e_index], length)
            e.append(ed)

            index += 1

    print('number of edges:', len(e))

    # 读取边的连接关系
    with open(path+'edges_connect.txt', 'r') as f:
        for line in f.readlines():
            attrs = line.rstrip().split(',')
            eid = int(attrs[0])
            lefts = attrs[1].rstrip().split(' ')
            rights = attrs[2].rstrip().split(' ')

            for l in lefts:
                if len(l)>0: #确保非空
                    e[eid].left.append(int(l))
            for r in rights:
                if len(r)>0:
                    e[eid].right.append(int(r))

    g = Graph(v, e, v_num, e_num)

    print(mbr)

    return g, mbr


# 创建边连接文件，存储每条边的左邻居和右邻居，传入原来的边集文件路径以及存储的文件路径
def create_edges_connect(read_path, write_path):

    edges = []

    with open(read_path, 'r') as f:
        e_num = int(f.readline())
        for line in f.readlines():
            edges.append(line.rstrip().split(' '))

    with open(write_path, 'w') as f:
        for i in tqdm(range(len(edges))):
            left = []
            right = []
            # 本路段id、left、right用逗号隔开，然后left以及right的id列表用空格隔开
            f.write('{},'.format(i))
            for j in range(len(edges)):
                # 有向边的话，同入同出应该不算？
                if edges[i][0] == edges[j][1]:
                    left.append(j)
                if edges[i][1] == edges[j][0]:
                    right.append(j)
                #if edges[i][1] == edges[j][1]:
                #    right.append(j)
                #if edges[i][0] == edges[j][0]:
                #    right.append(j)
            for l in left:
                f.write('{} '.format(l))
            f.write(',')
            for r in right:
                f.write('{} '.format(r))
            f.write('\n')



