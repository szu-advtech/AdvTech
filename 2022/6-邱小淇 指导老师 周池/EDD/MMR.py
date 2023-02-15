"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""

import copy
from math import inf

from EDDVertex import VerMenforEDD
from Edda import dijkstra, printEDDaTree, printEDDCost
from Graph import Graph
from Path import Path


def alogrithm2_GenertateTc(graph):
    #记录到每个节点的代价，
    #记录每个节点是否访问
    #记录每个节点的父节点
    visitedset= {}
    visitedset[0]=(VerMenforEDD(id=0,depth= -1, cost=0,parent=-1))
    Treec=Graph()
    costc = 0
    Treec.addVertex(0)
    ifchoosed_tmeps=False
    print('当前进入 MMR2 的子算法 先生成Tc树 \n 原始图信息为：')
    print(graph)
    while not graph.judgeFullVisitedR():
        mincost = inf
        (first, second, w) = (-1, -1, 0)
        isR1=True
        isR2=True
        #每一次while 都会增加一个R节点
        #遍历已经访问过的所有节点
        #print('当前访问的节点信息为：')

        for id,vd in visitedset.items():
            #print(vd) 打印每个eddaformen节点的信息，就是parent fromcost，depth等

            curvid = vd.id
            curvcost= vd.fromcost

            if curvid ==0 and ifchoosed_tmeps == False:
                ifchoosed_tmeps=True
                #找到第一个r节点
                for vid,v in graph.Gvertexlist.items():
                    if v.isR and graph.getVisited(vid) == False:#没访问过切实r节点
                        if curvcost + graph.y < mincost:
                            mincost=curvcost+graph.y
                            first,second,w = curvid,vid,graph.y

            elif curvid !=0:
                curv=graph.getVertex(curvid)
                curvadj=curv.getNeighbors()
                for vid,w1 in curvadj.items():
                    if graph.getVisited(vid) == False and graph.Gvertexlist[vid].isR:
                        if curvcost + w1 < mincost:
                            mincost =curvcost + w1
                            first,second,w=curvid,vid,w1



            else:
                continue
        ### 已经找到了 最小的可添加的边
        print('选中的边为：' +str(first)+' - '+ str(second)+' weight: '+str(w))
        if second == -1:
            besthortpath = []
            minlenpath = inf
            # 上述邻居不成功
            for si in Treec.Rservers:
                for sj in graph.Rservers:
                    if sj != si and graph.getVisited(sj) == False:
                        print('si  sj:' + str(si) + '  ' + str(sj))
                        pathtemp = dijkstra(graph, si, sj)
                        if minlenpath > pathtemp.cost:
                            minlenpath = pathtemp.cost
                            besthortpath = copy.copy(pathtemp)
            parent = -1
            for ivd in besthortpath.path:
                if parent == -1:
                    parent = ivd
                    continue

                if graph.getVisited(ivd) == False:
                    first = parent
                    second = ivd
                    w = graph.findEdgeWeight(first, second)
                    isR1 = graph.Gvertexlist[first].isR
                    isR2 = graph.Gvertexlist[second].isR
                    # 设置标记
                    graph.setVistexVist(second, True)  # 设置访问标记符号
                    vseco = VerMenforEDD(id=second, depth=visitedset[first].depth + 1,
                                         cost=visitedset[first].getCost() + w, parent=first)
                    visitedset[second] = vseco  # 已经访问的点加入到setvisited中
                    Treec.addVertex(second)  # 加入结果树中
                    Treec.setR(second, isR2)
                    Treec.addVertex(first)
                    Treec.setR(first, isR1)
                    if (Treec.addEdge(first, second, w)):
                        costc += w
            second = -1
        if second !=-1:
            graph.setVistexVist(second,True) # 设置访问标记符号
            vseco = VerMenforEDD(id=second, depth=visitedset[first].depth+1, cost=visitedset[first].getCost()+w ,parent=first)
            visitedset[second] = vseco # 已经访问的点加入到setvisited中
            Treec.addVertex(second) # 加入结果树中
            Treec.setR(second,isR2)
            Treec.addVertex(first)
            Treec.setR(first,isR1)
            if(Treec.addEdge(first,second,w)):
                costc += w

     # 结束
    print('当前SMT树（Tc）的总代价为： '+ str(costc) +' ； Tc树 的图结构如下：')
    print(Treec)
    return Treec,visitedset
def printSetVisited(visitset):
    print('当前的mmr 访问过的节点信息是：')
    for vid,v  in visitset.items():
        print(v)
def exeMMR(graph):
    graph.addVertex(0)
    for i in range(1,graph.n):
        graph.addEdge(0,i,graph.y)
    graph.sortGfrom1ToN()
    #print(graph)
    tc, visitedset = alogrithm2_GenertateTc(graph)
    # 开始调整违规的边
    #每一条边都加入到alogtithm中:
    printSetVisited(visitedset)
    for vid,vattitution in visitedset.items():
        #print('当前访问的节点： '+str(vid))
        #print(visitedset[vid].setvisited(True))
        if vattitution.getDepth()>=graph.dlimit:
            #违背了最低延迟限制
            path=dijkstra(graph,start=0,end=vid)
            #print(path)
            #我们这种情况下 违背最低延迟限度的情况下，mmr 通过dijsktra得到的结果都是 结果直接连接cloud。
            #把path 上的点加入到整个tc上，并删除整原始的边
            parent=vattitution.getpartent()
            tc.delEdge(parent,vid)
            if len(tc.Gvertexlist[parent].getNeighbors())==0:
                tc.delVertex(parent)
            if len(tc.Gvertexlist[vid].getNeighbors()) ==0:
                tc.delVertex(vid)

            bf = path.start
            af = 0
            #print('当前更新的路径为： '+str(path))
            for i  in range(1,len(path.path)):
                af=path.path[i]
                tc.addVertex(bf)
                tc.addVertex(af)
                wb=graph.findEdgeWeight(bf,af)
                tc.addEdge(bf,af,wb)
                #print(str(bf)+' - '+str(af))
                afvertex=VerMenforEDD(id=af,depth=visitedset[bf].depth+1,cost=visitedset[bf].getCost()+1,parent=bf)
                visitedset[af]=afvertex
                bf = af
            visitedset[path.path[-1]].setvisited(True)
    #结束
    print('\nMMR 算法结束，调整后的Tc数 的图结构如下：')
    print(tc)

    printSetVisited(visitedset)

    costall=printEDDCost(tc)
    print('MMR计算得到的输出代价为：' + str(costall) +' ；调整后的Tc树的连接关系为：（）为同一层节点')
    printEDDaTree(tc,visitedset)

    return costall
"""
TC:                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
G={
0	 adjacency:[(1, 9), (6, 9), (7, 9), (9, 9)]	 connecty: 4	 visited: False	 isR: False	 isT_ms: False,
1	 adjacency:[(0, 9), (2, 9), (3, 9), (4, 9)]	 connecty: 4	 visited: False	 isR: False	 isT_ms: False,
2	 adjacency:[(1, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
3	 adjacency:[(1, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
4	 adjacency:[(1, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
6	 adjacency:[(0, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
7	 adjacency:[(0, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
9	 adjacency:[(0, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
}
visitedset:
当前的mmr 访问过的节点信息是：
vid: 0	vdepth: -1	vfrocost: 0	vparent: -1	visited: True
vid: 1	vdepth: 0	vfrocost: 9	vparent: 0	visited: True
vid: 2	vdepth: 1	vfrocost: 18	vparent: 1	visited: True
vid: 3	vdepth: 1	vfrocost: 18	vparent: 1	visited: True
vid: 4	vdepth: 1	vfrocost: 18	vparent: 1	visited: True
vid: 6	vdepth: 0	vfrocost: 1	vparent: 0	visited: True
vid: 7	vdepth: 0	vfrocost: 1	vparent: 0	visited: True
vid: 9	vdepth: 0	vfrocost: 1	vparent: 0	visited: True
 """

