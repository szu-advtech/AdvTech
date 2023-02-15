"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""

import queue

class VerMenforEDD:
    def __init__(self,id=0,depth=0,cost=-1,parent=-1):
        self.id = id
        self.depth = depth
        self.fromcost = cost
        self.parent=parent
        self.visited = False
    def setvisited(self,g=True):
        self.visited=g
        return self.visited

    def setId(self,id):
        self.id=id
    def getId(self,id):
        return  self.id
    def setDepth(self,d):
        self.depth=d
    def setCost(self,c):
        self.fromcost=c
    def getDepth(self):
        return self.depth
    def getCost(self):
        return self.fromcost
    def setpartent(self,p):
        self.parent=p
    def getpartent(self):
        return self.parent

    def __str__(self):
        res = 'vid: '+str(self.id)+\
            '\tvdepth: '+str(self.depth)+\
            '\tvfrocost: '+str(self.fromcost)+\
            '\tvparent: '+str(self.parent)+\
            '\tvisited: '+str(self.visited)
        return res
def printEDDV(tedaaaver):
    print('EDDA 当前的所有节点状态如下：')
    for vid,eddv in tedaaaver.items():
        print(eddv)
    print('\n')

def findParent(tms,id):
    #BPF 找parent
    q=queue.Queue()
    q.put(0)
    visit=[]
    parent =-1
    while not q.empty():
        curid=q.get()
        adj= tms.getVertex(curid).getNeighbors()
        visit.append(curid)
        for vid ,w in adj.items():
            if vid not in visit:
                if vid == id:
                    parent = curid
                    break
                q.put(vid)
                visit.append(vid)
    return parent