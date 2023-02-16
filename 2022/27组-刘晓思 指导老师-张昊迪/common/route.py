
class Route:
    '''
    路径：由路网中的连续的若干条边构成
    参数：路径id，边的序列（有序）
    '''
    def __init__(self, id, start=None, end=None, e_list=[]):
        self.id = id
        self.e_list = e_list
        self.start = start   # 路径的开始点（Vertex类型）
        self.end = end     # 路径的结束点

    '''
    向路径中添加边 传入一条路网边（Edge类型）
    1. 判断本路径边集是否为空，是的话直接添加，改变start和end点
    2. 不为空，判断首尾相接情况，看将这条边放在路径的起始还是末尾，更新对应的start或者end
    考虑到路径和边是有向的，因而（start，end），判断r的start/end对应与e的end/start是否相等，相等则添加到前/后，否则提示无法添加
    '''
    def add_edge(self, e):
        # 当前路径为空，直接添加
        if len(self.e_list) == 0:
            self.e_list.append(e)
            self.start = e.start
            self.end = e.end
        else:
            # print(type(e.end))
            # print(type(self.start))
            if self.start == e.end:
                # 在路径前添加
                self.e_list.insert(0, e)
                # 更新start节点
                self.start = e.start
            elif self.end == e.start:
                # 在路径后添加
                self.e_list.append(e)
                # 更新end节点
                self.end = e.end
            else:
                # print('Route add error！')
                return False
        return True

