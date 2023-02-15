"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""

import  copy
from math import inf

import numpy as np
mincost=inf
solution={}
def checkout(graph,Tvisit):
    for vid,v in graph.Gvertexlist.items():
        if v.isR and Tvisit[vid-1] == False:
            return False
    return  True
def fillout(graph, path, tvist, transd, cost):
    if transd >= graph.dlimit:
        return
    copyTvisit=copy.copy(tvist)

    #print('当前comb下的 path：')
    #printSolution(path)

    #所有当前访问过的节点的相连边
    for i in range(len(tvist)):
        if tvist[i] ==1:
            vertexVisit=graph.getVertex(i+1)
            for vid,w in vertexVisit.getNeighbors().items():
                if copyTvisit[vid-1] ==0 and graph.Gvertexlist[vid].isR:
                    copyTvisit[vid-1]=1

                    path[i+1].append((vid,w))
                    path[vid].append(((i+1),w))
                    cost += w
    #print('当前comb下的fillout 一次之后的 path：')
    #printSolution(path)
    #print('当前路径得到的总价值： ' + str(cost))
    isfini=checkout(graph,copyTvisit)
    if isfini:
        global mincost
        global solution
        if cost < mincost:
            mincost = cost
            solution = copy.deepcopy(path)
        return
    #没结束就 下一层遍历
    fillout(graph,path,copyTvisit,transd+1,cost)

def findpath(graph, comb, lenforcomb):
    #
    Tn=len(graph.Gvertexlist)
    Tvisit=[]
    path = {}
    path[0]=[]
    for i in range(Tn):
        Tvisit.append(0)
        path[i+1]=[]

    cost=0
    for i  in range(lenforcomb):
        idv=comb[i]
        Tvisit[comb[i]-1]=1
        path[0].append((idv,graph.y))
        path[idv].append((0,graph.y))
        cost +=graph.y

    if checkout(graph,Tvisit):
        global mincost
        global solution
        if cost < mincost:
            mincost=cost
            solution=copy.deepcopy(path)
            #printSolution(solution)
            #print('当前路径得到的总价值： ' + str(mincost))
    #print('find path 当前的comb 为 ：'+ str([u for u in comb]) +' 准备执行填充完到所有R节点的路径 fillout')
    path = fillout(graph,path,Tvisit,1,cost)
    return path
#last 是新加入vid的编号
def Cloud2EdgesAdd(graph, comb, last, lenforcomb):
    copycomb=copy.copy(comb)
    copycomb.append(-1) #添加一个长度
    #print('当前节点是combination last：'+str(last) +' len of copycomb $ comb: '+str(len(copycomb))+' '+str(lenforcomb))
    #print(copycomb)
    findpath(graph,comb,lenforcomb)
    vertex=graph.Gvertexlist
    for vid,v in vertex.items():
        #设置里 vid是按照从小到大排序的 1，2，3，...
        if vid > last:
            #copycomb的长度是10，last2
            copycomb[lenforcomb]=vid
            Cloud2EdgesAdd(graph, copycomb, vid, lenforcomb + 1)
            #每一次都是在一条线上往深处去累加 一个直连cloud的点，知道findpath 函数推出，整个combinations才会结束
    return copycomb
def printSolution(solutiondict):
    #Tn=len(solutiondict)
    #Visit=[0 for i in range(Tn)]
    for vid,adj in solutiondict.items():
        if len(adj) ==0 :
            continue
        res = str(vid) +' - ('+'- '.join(str(i)for (i,w) in adj) + ')'
        print(res)

def exeEDDIP(graph):
    comb=[]
    comb.append(-1)
    vertex=graph.Gvertexlist
    for vid,v in vertex.items():
        #第一个节点默认和cloud相连
        #print('当前启动的cloud-vertex系列是： '+str(vid))
        comb[0]=vid #1
        #节点必须1-2-3-。。。顺序排序
        Cloud2EdgesAdd(graph, comb, vid, 1)

    print('EDDIP方法的总价值为：'+str(mincost)+"; 找到的传输方法为：")

    printSolution(solution)
    return mincost