
class Point:
    '''

    轨迹点：经纬度

    '''
    def __init__(self, lat, lng):
        self.lat = lat
        self.lng = lng

    # 定义字符串输出方法（print）
    def __str__(self):
        return '({},{})'.format(self.lat, self.lng)

    # 重写 == 方法，判断坐标是否一致
    def __eq__(self, other):
        return self.lat == other.lat and self.lng == other.lng
