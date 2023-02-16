"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""

import queue
import random

#当传的是字典型(dictionary)、列表型(list)时,对其进行操作（修改值，添，改，删），则会改变
#不可变对象：Number, String, Tuple, bool
#可变对象：List, Set, Dict 是可以改变内部的元素
def BDFToNeighbor(graph,nodeid,rundepth):
    tempcost=0

    #访问节点nodeid的是r的邻居节点，放入到queue，BPF优先遍历
    vertex = graph.getVertex(nodeid)
    queueR = queue.Queue(graph.r)
    vertexAdj=vertex.adjcency  #字典 【id ：w】
    for ovid,w in vertexAdj.items():
        ov=graph.getVertex(ovid)
        if ov.isVisited == False and ov.isR:
            tempcost += w
            queueR.put(ov.id)
            ov.isVisited=True
            print(str(nodeid) +' <-> '+str(ovid))

    if rundepth < graph.dlimit-1 :
        #print('dan ' +str(rundepth) + str(queueR.empty()))
        while(queueR.empty()==False):
            nextid = queueR.get() #get 直接弹出
            #print('nextid : '+ str(nextid))
            tempcost += BDFToNeighbor(graph,nextid,rundepth+1)

    return tempcost
def Greedy(graph):
    isfinished = False
    graph.resetGVisited()
    gcost=0

    while(isfinished == False):
        maxId= graph.findmaxConnect()
        graph.setVistexVist(maxId)
        res = BDFToNeighbor(graph,maxId,1)
        if res !=0 or graph.getVertex(maxId).isR:
            gcost += graph.y
            gcost +=res
            #print(str(gcost) +' ss ' + str(maxId))
        isfinished=graph.judgeFullVisitedR()

        #false 继续循环
        #true 退出循环

    return gcost

def exeGreedy(graph):
    print('贪心策略 传输路径')
    cost=Greedy(graph)
    print('贪心策略 传输代价：'+ str(cost))
    return cost
def Random(graph):
    isfinished = False
    graph.resetGVisited()
    rcost=0
    V= graph.n
    while(isfinished == False):
        cloudId=random.randint(1,int(V)) # [1,V]
        #print(cloudId)
        graph.setVistexVist(cloudId)
        res = BDFToNeighbor(graph,cloudId,1)
        if res !=0 or graph.getVertex(cloudId).isR:
            rcost += graph.y
            rcost +=res
        isfinished = graph.judgeFullVisitedR()

        # false 继续循环
        # true 退出循环
    return rcost
def exeRandom(graph):
    print('随机策略 传输路径')
    cost=Random(graph)
    print('随机策略 传输代价：'+ str(cost))
    return cost
