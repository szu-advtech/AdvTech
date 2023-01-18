"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""

from math import inf
from copy import copy, deepcopy


class Path:
    def __init__(self,s=-1,cost=inf):
        self.path = []
        self.cost =cost  #该条路径的总代价
        self.start=s
        if s!=-1:
            self.path.append(self.start)
    def __str__(self):
        res = '当前路径的总代价为：'+str(self.cost) +'\n'+ \
            str('-'.join(str(k) for k in self.path))
        return res
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    def __deepcopy__(self, memo={}):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        for k, v in self.__dict__.items():
            setattr(result, k, deepcopy(v, memo))
        return result
    #设置当前cost 最低的优先级最高
    def __lt__(self, other):
        return self.cost < other.cost

    def addToPath(self,id,w):
        if id not in self.path:
            self.path.append(id)
            self.cost+=w
            return True
        else:
            return False
    def isEmpty(self):
        if len(self.path) == 0:
            return True
        else:
            return False

    def getPathend(self):
        if not self.isEmpty():
            return self.path[-1]

    def getPathstart(self):
        if not self.isEmpty():
            return self.path[0]