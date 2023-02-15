class block:
    tupledict = {}
    mass = 0.0
    size = 0.0
    dimension = 0
    colsetDict = {}
    colDegreeDicts = {}
    colKeysetDicts = {}

    def __init__(self, tupledict,  colsetDict, colDegreeDicts, colKeysetDicts, mass, size, dimension):
        self.tupledict = tupledict
        self.mass = mass
        self.size = size
        self.dimension = dimension
        self.colsetDict = colsetDict
        self.colDegreeDicts = colDegreeDicts
        self.colKeysetDicts = colKeysetDicts

    #添加新项
    #需要做的工作有：
    #1、添加块的质量 2、添加到快的tupledict 3、若属性值是新的，则块的size需要添加，
    # 块该维度新属性的值需要更新，colkeysetDicts的二维需要新建一项。否则，只需要更新colDegreeDicts和colKeysetDicts
    def addUpdate(self, changetuples):
        for key in changetuples:
            value = changetuples[key]
            if key not in self.tupledict:
                self.tupledict[key] = value
                self.mass = self.mass + value
                for i in range(self.dimension):
                    attr = key[i]
                    if attr not in self.colsetDict[i]:
                        self.size = self.size + 1
                        self.colDegreeDicts[i][attr] = value
                        self.colsetDict[i].add(attr)
                        self.colKeysetDicts[i][attr] = set()
                        self.colKeysetDicts[i][attr].add(key)
                    else:
                        self.colDegreeDicts[i][attr] += value
                        self.colKeysetDicts[i][attr].add(key)

    #移除某一项
    #所需进行的工作 1、tupledict弹出对应的key和value 2、对应块的mass减少
    # 3、对应项的colDegreeDict减少，colkeysetdict去除该项，去除完后检查值是否等于0。
    #若等于0则去除相应的项，对应块的size也应减少
    def removeUpdate(self, changetuples):
        for key in changetuples:
            value = changetuples[key]
            self.tupledict.pop(key)
            self.mass -= value
            for i in range(self.dimension):
                attr = key[i]
                self.colDegreeDicts[i][attr] -= value
                self.colKeysetDicts[i][attr].remove(key)
                if self.colDegreeDicts[i][attr] == 0:
                    self.colDegreeDicts[i].pop(attr)
                    self.colKeysetDicts[i].pop(attr)
                    self.colsetDict[i].remove(attr)
                    self.size -= 1

    def getTuples(self):
        return self.tupledict

    def getMass(self):
        return self.mass

    def getSize(self):
        return self.size

    def getDensity(self):
        return self.mass / self.size

    def getAttributeDict(self):
        return self.colsetDict

    def getColDegreeDicts(self):
        return self.colDegreeDicts

    def getColKeysetDicts(self):
        return self.colKeysetDicts