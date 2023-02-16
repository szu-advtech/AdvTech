"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""

class Vertex:
    def __init__(self,id):
        self.id = id
        self.con = 0 #记录该节点的度 大小
        self.adjcency = {}  #记录 连边的 相连节点 以及边上权重 ，othrev：weight
        self.isR =False #标记节点 是否需要接受 source 从cloud
        self.isVisited=False #是否被访问/root 访问过
        self.mst = False
    def getId(self):
        return self.id

    def getNeighbors(self):
        return self.adjcency

    #添加带权重的边 默认是权重1
    def addNeighbor(self,neighbor,weight=1):
        if neighbor not in self.adjcency.keys():
            self.adjcency[neighbor]=weight
            self.con +=1
            return True
        return False
    #返回 节点的度大小
    def getcon(self):
        return self.con
    #删除 指定节点的连边
    def delNeighbor(self,neighbor):
        if neighbor in self.adjcency.keys():
            del self.adjcency[neighbor]
            self.con -=1
            return True
        return False

    def setR(self,isR=False):
        self.isR=isR

    def setVisited(self,g=True):
        self.isVisited=g
    def getVisited(self):
        return self.isVisited
    #打印节点的 连边
    def __str__(self):
        res= str(self.id) + \
            '\t adjacency:' +str( [(k,w) for (k,w) in self.adjcency.items()]) + \
            '\t connecty: '+str(self.con) +\
            '\t visited: ' +str(self.isVisited) +\
            '\t isR: ' +str(self.isR)+\
            '\t isT_ms: ' +str(self.mst)
        return  res
class Graph:
    # 这个类包含二维的
    # n 节点个数 edges，con 是每个节点的 度大小，
    #R 节点是否是数据来自cloud/最后访问的节点覆盖 R 中所有节点就可以结束
    # Rnum 个数
    # graph 是邻接表 记录图的连接关系。
    #S 0 1  记录每个节点是否被访问过。 1 访问，0 未访问
    def __init__(self,y=20,d=2):
        self.n = 0
        self.e = 0
        self.Gvertexlist = {} #邻接表 ,id - vertex
        self.r = 0
        self.Rservers=[]
        self.Vertexcon={} #每个节点的度
        self.y= y # cloud to server cost
        self.dlimit =d #server to server d limit d=2。means  only one time
    def getVisited(self,vid):
        return self.Gvertexlist[vid].getVisited()
    #添加连边
    def addEdge(self,left,right,w=1):
        if self.Gvertexlist[left].addNeighbor(right,w) and self.Gvertexlist[right].addNeighbor(left,w):
            self.e +=2
            return True
        else:
            return False
    #删除连边
    def delEdge(self,left,right):
        if self.Gvertexlist[left].delNeighbor(right) and self.Gvertexlist[right].delNeighbor(left):
            self.e -=2
            return True
        return False
    #添加节点
    def addVertex(self,id):
        if id not in self.Gvertexlist.keys():
            self.Gvertexlist[id] = Vertex(id)  ###id  是key，vlaue是 vertex 结构点（id，con，adj，visited，R）
            self.n += 1
            return True
        return False

    def getVertex(self,id):
        return self.Gvertexlist[id]
    #删除节点
    def delVertex(self,id):
        if id in self.Gvertexlist.keys():
            del self.Gvertexlist[id]
            self.n -=1
            return  True
        return  False
    #指定 某个节点 是r节点
    def addR(self,id):
        self.r+=1
        if id not in self.Rservers:
            self.Rservers.append(id)
        self.Gvertexlist[id].setR(True)
    def setR(self,id,isR=False):
        if isR and id >0:
            self.addR(id)
        else:
            self.Gvertexlist[id].setR(isR)
    def setTms(self,id,isT=True):
        self.Gvertexlist[id].mst=True
    #判断vid是否在mst树种
    def getTms(self,id):
        return self.Gvertexlist[id].mst
    #对字典按照key从小到大排序
    def  sortGfrom1ToN(self):
        sortedList=sorted(self.Gvertexlist.items(),key=lambda item:item[0])
        self.Gvertexlist.clear()
        for id,vertex in sortedList:
            self.Gvertexlist[id]=vertex

    #打印 G
    def __str__(self):
        #res ='G= {\n ' +\
        #    str([v for k,v in self.Gvertexlist.items()]) + \
        #    '}\n'
        res = 'G={\n'
        for k,v in self.Gvertexlist.items():
            res +=str(v)+',\n'
        res +='}\n'
        return  res

    #辅助方法： 找到visited = false的maxconnect节点
    def findmaxConnect(self):
        maxconne=-1
        maxconid=-1
        for k,v in self.Gvertexlist.items():
            if maxconne < v.con and v.isVisited==False:
                maxconne=v.con
                maxconid=k
        return maxconid
    def resetGVisited(self):
        for k,v in self.Gvertexlist.items():
            v.isVisited=False
    def setVistexVist(self,vid,g=True):
            self.Gvertexlist[vid].setVisited(g)
    def judgeFullVisitedR(self):
        for k,v in self.Gvertexlist.items():
            if v.isR and v.isVisited == False:
                return False

        return True
    def findEdgeWeight(self,s,vs):
        #v=
        sadj=  self.Gvertexlist[s].getNeighbors()
        for v,w in sadj.items():
            if v==vs:
                return w
        return  False



