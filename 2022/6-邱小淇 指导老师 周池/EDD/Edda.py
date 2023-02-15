"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""

from math import inf

from EDDVertex import VerMenforEDD, findParent, printEDDV
from Graph import Graph
import queue
from copy import copy, deepcopy
from Path import Path
from Stack import Stack
#不设置层数限制的递归
def BEFToNeighbor(graph,mst,nodeid):
    tempcost =0
    #把与node相连的是r节点的所有结点放入到 queue中并设置tms
    vertex = graph.getVertex(nodeid)
    queueR = queue.Queue(graph.r)
    vertexAdj = vertex.adjcency  # 字典 【id ：w】
    mst.addVertex(nodeid)
    mst.setR(vertex.id,vertex.isR)
    #print(vertex.id)
    for ovid, w in vertexAdj.items():
        ov = graph.getVertex(ovid)
        if ov.isVisited == False and ov.isR:
            #print(ov.id)
            tempcost += w
            #graph.setVistexVist(ovid)
            ov.isVisited = True
            ov.mst=True
            #放入mst
            mst.addVertex(ovid)
            mst.addEdge(nodeid,ovid,w)
            mst.setR(ov.id,ov.isR)
            mst.setTms(ov.id)
            #放入队列
            queueR.put(ov.id)
            print(str(nodeid) + ' <-> ' + str(ovid))
    while(queueR.empty()==False):
        next=queueR.get()
        #print('在DFS 加入边节点中，当前的要XX点的相邻边 '+str(next))
        tempcost += BEFToNeighbor(graph,mst,next)

    return tempcost
def cmst(graph):
    print('现在EDDA 先创建CMST 树')
    mst = Graph(9,2)
    isfinished=False
    cost=0
    while(isfinished==False):
        maxconid=graph.findmaxConnect()
        if maxconid ==-1:
            print(graph)
            break
        mst.addVertex(0)
        mst.addVertex(maxconid)
        mst.addEdge(0,maxconid,graph.y)
        #设置访问
        #graph.setVistexVist(maxconid)
        graph.Gvertexlist[maxconid].isVisited=True
        graph.Gvertexlist[maxconid].mst=True
        tcost=BEFToNeighbor(graph,mst,maxconid)
        if tcost!=0:
            cost += tcost
            cost += graph.y
        isfinished = graph.judgeFullVisitedR()

    print("ＣＭＳＴ的传输代价："+ str(cost) +' ； CMST树的 图结构如下：')
    print(mst)
    return mst
"""
这里输出一条dijstra 的路径输出：
当前路径的总代价为：3
1-3-6-9
"""
def dijkstra(graph, start,end):
    que = queue.PriorityQueue() # 堆排序求最优

    temp = Path(start,0)
    #que.put([temp.cost,temp])
    que.put(temp)
    #print('进入dijkstr算法  strart: '+str(start)+' end: '+str(end))
    while(que.empty()==False):

        bestpath = que.get()

        #print(bestpath)
        lastoneid=bestpath.path[-1]
        if lastoneid == end:
            return  bestpath #返回path
        else:
            vertex = graph.getVertex(lastoneid)
            vertexadj = vertex.getNeighbors()
            for v,w in vertexadj.items():
                producepath = deepcopy(bestpath)
                if producepath.addToPath(v,w):
                    que.put(producepath)

"""printEDDTreeRoot:
0-1 - (2 - 3 - 4)
0-7 - (6)
0-9
1-2
1-3
1-4
7-6"""
def printEDDTreeRoot(tedda,teddvertex,root):
    tedda.getVertex(root).setVisited()
    vertexadj = tedda.getVertex(root).getNeighbors()
    for vidfromcloud, w in vertexadj.items():

        if tedda.getVertex(vidfromcloud).getVisited():
            continue
        vchildadj = tedda.getVertex(vidfromcloud).getNeighbors()
        res = str(root) +'-'+ str(vidfromcloud)
        tems = ''
        Senc = False
        bp = 0
        lp = 0
        for vid, w in vchildadj.items():

            if  vid == root or vid ==0:
                continue

            if not Senc :
                bp = teddvertex[vid].getpartent()
            else:
                lp = teddvertex[vid].getpartent()
            # 这是为了兼顾 edda树有三层或者以上的输出情况，同意括号是同一层的节点
            if bp != lp:
                if Senc:
                    tems += ')'
                    res += tems
                tems = ' - (' + str(vid)
            else:
                tems += ' - ' + str(vid)
            if not Senc:
                Senc = True
            else:
                bp = lp
        if tems != '':
            tems += ')'
        print(res + tems)

def printEDDaTree(tedda,teddvertes):
    tedda.resetGVisited()
    que = queue.Queue()
    que.put(0)
    while(que.empty()==False):
        root = que.get()
        vertexadj = tedda.getVertex(root).getNeighbors()
        if not tedda.getVertex(root).getVisited():
            printEDDTreeRoot(tedda,teddvertes,root)
        #把子节点也放进去：
        for v,w in vertexadj.items():
            if not tedda.getVertex(v).getVisited():
                que.put(v)

def printEDDCost(tedda,name='EDDA'):
    cost =0

    vertexadj = tedda.getVertex(0).getNeighbors()
    for vidfromcloud,w in vertexadj.items():
        vchildadj=tedda.getVertex(vidfromcloud).getNeighbors()
        for vid,w in vchildadj.items():
            cost += w
            if vid == 0:
                continue

    return  cost

def edda(graph,tms):
    tedda = tms
    teddaver= {}
    """
    for vid,v in graph.Gvertexlist.items():
        tv = VerMenforEDD()
        tv.setId(vid)
        if graph.getTms(vid): #是 tms 树的节点
            p = findParent(tms,vid)
            tv.setpartent(p)
            if p==0:

                tv.setCost(graph.y)
                tv.setDepth(0)
            else:
                if p in teddaver.keys():
                    tv.setCost(1 + teddaver[p].getCost())
                    tv.setDepth(1 + teddaver[p].getDepth())
        else:
            tv.setpartent(0)
        teddaver[vid]=tv
    printEDDV(teddaver)
    """
    #设置初始的父节点 当前节点的访问代价，深度
    for vid,v in graph.Gvertexlist.items():
        tv = VerMenforEDD()
        tv.setId(vid)
        if graph.getTms(vid): #是 tms 树的节点
            p = findParent(tms,vid)
            tv.setpartent(p)
        #else:
        #    tv.setpartent(0)
        teddaver[vid]=tv
    #printEDDV(teddaver)
    # 防止父节点出现的顺序在teddaver中是 当前节点之后
    for vid,v in teddaver.items():
        p = teddaver[vid].getpartent()
        if  p!= -1: #找到父节点的编号了
            if p ==0:
                v.setCost(graph.y)
                v.setDepth(0)
            else:
                v.setCost(1+teddaver[p].getCost())
                v.setDepth(1+teddaver[p].getDepth())
        else:
            v.setCost(inf)
            v.setDepth(0)
    printEDDV(teddaver)

    #清空graph的所有节点的访问状态
    graph.resetGVisited()
    stack = Stack()
    stack.put(0) #云节点
    while not stack.isEmpty():
        vid = stack.peek()
        stack.pop()
        vertex = tms.getVertex(vid)
        vertexadj = vertex.getNeighbors()
        #print('当前节点是：'+str(vid))
        if vid == 0:
            for v,w in vertexadj.items():
                if not graph.getVertex(v).isVisited:
                    stack.put(v)
            graph.addVertex(0)
            graph.setVistexVist(0,True)

            #print(stack)
            #print(graph)
            continue
        #只有在这里会设置 节点的访问为 True
        graph.setVistexVist(vid)
        teddaver[vid].setvisited()
        #获取到当前节点的相连节点
        # print(teddaver[vid].getDepth())
        # print(vertex.isR)

        if vertex.isR == False or teddaver[vid].getDepth()<graph.dlimit:
            #print('进入第二种情况')

            for v,w in vertexadj.items():

                if not graph.getVisited(v):
                    teddaver[v].setpartent(vid)
                    stack.put(v)

            #print(stack)
            #print(graph)
            continue
        #不满足dlimit的 r节点 tms中的。
        #print(stack)
        #print(graph)
        #printEDDV(teddaver)
        #进入第三种情况
        #print('第三中情况，当前处理节点：(不符合dlimit限制得r节点) ' + str(vid))
        mincost = inf
        tpath=Path(0)
        #找到和 c直连的点的 到当前vid 节点的最小路径
        cloudv =tms.getVertex(0)
        cloudadj=cloudv.getNeighbors()
        for vs,w in cloudadj.items():
            #vs是直接连接cloud云节点的 节点，计算c-vs-vid（当前节点）的最小路径
            #print('进入找到从cloud 经过未知节点vs 到达vid 的最短路径：输出最短路径')
            tempath = dijkstra(graph,vs,vid)
            #tempath.cost=tempath.cost+graph.y
            #print(tempath)
            #打印找到的当前路径

            if not tempath:
                continue


            if  len(tempath.path)!=0 and tempath.cost<mincost:
                mincost=tempath.cost
                tpath=tempath

        #找到最短路径后 比较 cost 的最大限制
        # 找到tms数目之内可以自己更新的节点了
        #print('mincost: '+str(mincost))
        if(mincost < graph.dlimit):
            #print(str(vid)+' 节点 在tms节点之间可以找到一个合法得可行路径')
            for i  in range(len(tpath.path)):
                curvid=tpath.path[i]
                if i==0: #起始节点一定是和cloud联通的吗 ？ 这这里是的 因为tpath 是通过 和cloud直接联通的 vs获得的
                    teddaver[curvid].setpartent(0)
                    teddaver[curvid].setCost(graph.y)
                    teddaver[curvid].setDepth(0)
                    continue
                parentid=tpath.path[i-1] #路径上前一个点就是它的父亲节点
                teddaver[curvid].setpartent(parentid)
                wcost=graph.findEdgeWeight(parentid,curvid)
                teddaver[curvid].setCost(teddaver[parentid].getCost()+wcost)
                teddaver[curvid].setDepth(teddaver[parentid].getDepth()+1)
                #添加边 如果存在则不会重新添加
                tedda.addEdge(curvid,parentid)
            #更新tms树上的所有节点信息 parent cost depth
            for vidt,v in graph.Gvertexlist.items():
                if vidt != vid and v.isR:
                    t2path = dijkstra(graph,vid,vidt) #找到这个节点 到其他r节点的最短路径
                    #print(t2path) #打印找到的最短路径 c-？？XX- vid
                    if not  t2path:
                        continue
                    if not t2path.isEmpty():
                        if teddaver[vidt].getCost() - teddaver[vid].getCost() > t2path.cost:
                            #c从vid 到 vidt（其他r节点）的距离是 - 结果，当前找到一个跟路径就更新 vidt的cost
                            teddaver[vidt].setCost(teddaver[vid].getCost() + t2path.cost)
                            #print('调整边 修改父节点和cost ' + str(vid) +'->'+str(vidt))
                            #未path路上的每一个前后节点添加边到tedda
                            for i in range(1,len(t2path.path)): ##应该在这里添加路径的
                                v1=t2path.path[i] #当前节点
                                v2=t2path.path[i-1] #子节点
                                tedda.addVertex(v1)
                                tedda.addVertex(v2)
                                wcost=graph.findEdgeWeight(v1,v2)
                                tedda.addEdge(v1,v2,wcost)
                                teddaver[v1].setDepth(teddaver[v2].getDepth()+1)
                                teddaver[v1].setpartent(v2)
        else: #找到的路径 长度超限
            teddaver[vid].setpartent(0)
            teddaver[vid].setDepth(0)
            teddaver[vid].setCost(graph.y)
            tedda.addEdge(0,vid,graph.y)
        ## 调整完后 还需要更新 所有vid 的子节点信息。
        vertex = tedda.getVertex(vid)
        vertexadj=vertex.getNeighbors()
        for ov,w in vertexadj.items(): #6 ：3 7 9
            if ov==0:
                continue
            t3path = dijkstra(graph,vid,ov)
            if teddaver[ov].getCost()-teddaver[vid].getCost() > t3path.cost:
                teddaver[ov].setpartent(vid)
                wcost= graph.findEdgeWeight(ov,vid)
                teddaver[ov].setCost(teddaver[vid].getCost()+wcost)
                #print('更新 相连边的cost 父母'+str(vid) +'-'+str(ov))
            if teddaver[vid].getCost() - teddaver[ov].getCost() > t3path.cost:
                teddaver[vid].setpartent(ov)
                wcost = graph.findEdgeWeight(ov, vid)
                teddaver[vid].setCost(teddaver[ov].getCost() + wcost)
                #print('更新 相连边的cost 父母' + str(ov) + '-' + str(vid))
            #深度需要迭代去改
            for vidt,vedd in teddaver.items():
                if vidt == 0:
                    continue
                pp = teddaver[vidt].getpartent()
                if pp ==0:
                    teddaver[vidt].setDepth(0)
                elif pp!=-1 :
                    teddaver[vidt].setDepth(teddaver[pp].getDepth()+1)

        #循环处理结束 开始DFS查找
        for vidt,w in vertexadj.items():
            if vidt==0:
                continue
            if not graph.Gvertexlist[vidt].isVisited :
                stack.put(vidt)
    deledgeset=[]
    #while 循环结束
    #print('结束了 现在删除多余的边')
    #print(tedda)
    #printEDDV(teddaver)
    for vidt,v in tedda.Gvertexlist.items():
        for ov ,w in v.getNeighbors().items(): # v-ov1.v-ov2 vidt 的相邻节点中，父节点是vidt的则是要留下的边
            if ov==0 or vidt == 0:
                continue
            if  teddaver[ov].getpartent() != vidt and teddaver[vidt].getpartent() !=ov:
                if ((ov,vidt) not in deledgeset) and ((vidt,ov) not in deledgeset):
                    deledgeset.append((ov,vidt))
                else:
                    continue

    for edge in deledgeset:
        ov,vidt=edge[0],edge[-1]
        #print('现在删除的边 '+str(ov)+' - '+str(vidt))
        tedda.delEdge(ov,vidt)
        if tedda.getVertex(vidt).getcon() ==0:
            tedda.delVertex(vidt)
        if tedda.getVertex(ov).getcon() ==0:
            tedda.delVertex(ov)
    print("最后得TEDDA树  图结构如下： ")
    print(tedda)
    cost=printEDDCost(tedda)
    print( 'EDDA计算得到的传输代价为：' + str(cost) + '； EDDA树边的连接关系如下：（）为同一层节点')
    printEDDaTree(tedda,teddaver)
    return  tedda,cost
"""
最后得TEDDA 树： 
G={
0	 adjacency:[(1, 9), (7, 9), (9, 9)]	 connecty: 3	 visited: False	 isR: False	 isT_ms: False,
1	 adjacency:[(0, 9), (2, 1), (3, 1), (4, 1)]	 connecty: 4	 visited: False	 isR: True	 isT_ms: False,
2	 adjacency:[(1, 1)]	 connecty: 1	 visited: False	 isR: True	 isT_ms: True,
3	 adjacency:[(1, 1)]	 connecty: 1	 visited: False	 isR: True	 isT_ms: True,
4	 adjacency:[(1, 1)]	 connecty: 1	 visited: False	 isR: True	 isT_ms: True,
6	 adjacency:[(7, 1)]	 connecty: 1	 visited: False	 isR: True	 isT_ms: True,
7	 adjacency:[(0, 9), (6, 1)]	 connecty: 2	 visited: False	 isR: True	 isT_ms: True,
9	 adjacency:[(0, 9)]	 connecty: 1	 visited: False	 isR: True	 isT_ms: True,
}

EDDA计算得到的传输代价未：31
0-1 - (2 - 3 - 4)
0-7 - (6)
0-9
1-2
1-3
1-4
7-6
"""
def exeEDDA(g):

    mst=cmst(g)
    #print(mst)
    tedda,cost=edda(g,mst)
    return cost
