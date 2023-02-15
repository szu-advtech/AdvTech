"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""

import copy
import time
import networkx as nx
from matplotlib import pyplot as plt
##自定义的类
from CreateGraphGM import GM
from Edda import  exeEDDA
from Graph import Graph
from GreedyAndRandom import exeRandom, exeGreedy
from Eddip import exeEDDIP
from MMR import  exeMMR

### 建模：构建EDD问题场景，随机设置r节点，并可视化
def createGM(yg=20,dlimit=2,rnump=0.6,readfile='randomNetwork01.txt'):

    g= Graph(y=yg,d=dlimit)
    gshow = nx.Graph()
    try:
        rfile = open(readfile,mode='r')
        v=0
        SetV=True
        Vnum=0
        rnum=0
        vcolor=[]
        for line in rfile:
            #print(line)
            v+=1
            line = line.split(' ')

            # 生成点
            if SetV==True:
                Vnum = len(line) -1
                rnum = int(rnump * Vnum)
                for i in range(1, Vnum+1):
                    g.addVertex(i)
                    gshow.add_node(i)
                    vcolor.append('green')
                SetV=False
            #处理最后一行r节点
            if  v == Vnum+1:
                #最后一行
                lenr=len(line) -1
                print('r num:'+str(rnum))
                ser=[]
                for i in range(lenr):
                    if i < rnum:
                        g.addR(int(line[i]))
                        ser.append(line[i])
                        vcolor[int(line[i])-1]='red'
                if lenr < rnump:
                    #说明还不够：

                    for iv in range(1,1+Vnum):

                        if len(ser) == rnum:
                            break
                        if iv not in ser:
                            g.addR(iv)
                            ser.append(iv)
                            vcolor[iv-1]='r'
                print('R 节点序列：')
                ser.sort()
                print(ser)
                continue
            else:
                # 正常设置值
                for i in range(len(line)):
                    if line[i] == '1':
                        g.addEdge(v, i + 1, 1)
                        gshow.add_edge(v, i + 1)
    finally:
        rfile.close()
    print(g)
    nx.draw(gshow,with_labels=True,node_color=vcolor)
    plt.show()
    return g
def testCreatG():
    # 测试节点
    # v= Vertex(5)
    # v.addNeighbor(2)
    # v.addNeighbor(3)
    # print(v)
    # 测试图 y dlimit
    """
    # 第一个测试图
    # y是C2E传输代价，limit=2，E2E 传输跳数<2
    g = Graph(y=9,limit=2)
    for i in range(1, 11):
        g.addVertex(i)
    # 添加边 不加权重，默认为1
    g.addEdge(1, 2, 1)
    g.addEdge(1, 3, 1)
    g.addEdge(1, 4, 1)  # 3
    g.addEdge(2, 5)
    g.addEdge(2, 4)  # 5
    g.addEdge(3, 6)
    g.addEdge(3, 7)  # 7
    g.addEdge(4, 10)  # 8
    g.addEdge(5, 8)
    g.addEdge(6, 7)
    g.addEdge(6, 9)
    g.addEdge(7, 8)
    g.addEdge(8, 9)
    g.addEdge(9, 10)
    # 设置 R
    g.addR(1)
    g.addR(2)
    g.addR(3)
    g.addR(4)
    g.addR(6)
    g.addR(7)
    g.addR(9)
    :return:
    """

    """
    # 第二个测试图，节点个数为q,
    g = Graph(y=9,limit=2)
    for i in range(1, 11):
        g.addVertex(i)
    g.addEdge(1, 2, 1)
    g.addEdge(1, 5, 1)
    g.addEdge(3, 5, 1)
    g.addEdge(2, 3, 1)
    g.addEdge(3, 4, 1)
    g.addEdge(3, 6, 1)
    g.addEdge(8, 5, 1)
    g.addEdge(9, 5, 1)
    g.addEdge(4, 6, 1)
    g.addEdge(6, 7, 1)
    g.addEdge(9, 10, 1)
    g.addEdge(7, 10, 1)
    g.addR(1)
    g.addR(4)
    g.addR(5)
    g.addR(6)
    g.addR(9)
    g.addR(10)
    """
def exeRun():

    ### 正式测试： 先运行CreateGrapgGM.py,再运行main.py:createGM-> greedy/random/eddipo/edda/mmr
    g=createGM(yg=20,dlimit=2,rnump=0.6,readfile='randomNetwork01.txt')
    ### 删除程序中print 语句：print开销太大了
    cost=[]
    times=[]
    algorithmName=['Greedy','Random','EDDA','EDDIP','MMR']

    print('-------------- 执行：Greedy --------------')
    timeStart = time.time()  # 记录开始时间# function() 执行的程序time_end=time.time()# 记录结束时间time...
    c=exeGreedy(g)
    timeEnd = time.time()
    times.append(timeEnd-timeStart)
    cost.append(c)

    g.resetGVisited()
    print('\n-------------- 执行：Random --------------')
    timeStart = time.time()
    c=exeRandom(g)
    timeEnd = time.time()
    times.append(timeEnd - timeStart)
    cost.append(c)

    g.resetGVisited()
    geeda=copy.deepcopy(g)
    print('\n-------------- 执行：EDDA --------------')
    timeStart = time.time()
    c=exeEDDA(geeda)
    timeEnd = time.time()
    times.append(timeEnd - timeStart)
    cost.append(c)

    #g.resetGVisited()
    geddip=copy.deepcopy(g)
    print('\n---------------- 执行：EDDIP --------------')
    timeStart = time.time()
    c=exeEDDIP(geddip)
    timeEnd = time.time()
    times.append(timeEnd - timeStart)
    cost.append(c)
    #mmr的时候加入0-所有节点的边 已经g有了i需改
    #g.resetGVisited()
    gmmr=copy.deepcopy(g)
    print('\n----------------- 执行：MMR --------------')
    timeStart = time.time()
    c=exeMMR(g)
    timeEnd = time.time()
    times.append(timeEnd - timeStart)
    cost.append(c)

    print('\n\n 五种算法得到的cost 和 计算时间分别为：')
    for i in range(len(cost)):
        print(algorithmName[i]+': cost = '+str(cost[i])+' time = '+str(times[i])+' s')

if __name__ == '__main__':

    #testCreatG()
    exeRun()

"""
-------------- 执行：Greedy --------------
贪心策略 传输路径
1 <-> 2
1 <-> 3
1 <-> 4
3 <-> 6
3 <-> 7
8 <-> 9
贪心策略 传输代价：24

-------------- 执行：Random --------------
随机策略 传输路径
6 <-> 3
6 <-> 7
6 <-> 9
3 <-> 1
5 <-> 2
2 <-> 4
随机策略 传输代价：24

-------------- 执行：EDDA --------------
现在EDDA 先创建CMST 树
1 <-> 2
1 <-> 3
1 <-> 4
3 <-> 6
3 <-> 7
6 <-> 9
ＣＭＳＴ的传输代价：15 ； CMST树的 图结构如下：
G={
0	 adjacency:[(1, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
1	 adjacency:[(0, 9), (2, 1), (3, 1), (4, 1)]	 connecty: 4	 visited: False	 isR: True	 isT_ms: False,
2	 adjacency:[(1, 1)]	 connecty: 1	 visited: False	 isR: True	 isT_ms: True,
3	 adjacency:[(1, 1), (6, 1), (7, 1)]	 connecty: 3	 visited: False	 isR: True	 isT_ms: True,
4	 adjacency:[(1, 1)]	 connecty: 1	 visited: False	 isR: True	 isT_ms: True,
6	 adjacency:[(3, 1), (9, 1)]	 connecty: 2	 visited: False	 isR: True	 isT_ms: True,
7	 adjacency:[(3, 1)]	 connecty: 1	 visited: False	 isR: True	 isT_ms: True,
9	 adjacency:[(6, 1)]	 connecty: 1	 visited: False	 isR: True	 isT_ms: True,
}

EDDA 当前的所有节点状态如下：
vid: 1	vdepth: 0	vfrocost: 9	vparent: 0	visited: False
vid: 2	vdepth: 1	vfrocost: 10	vparent: 1	visited: False
vid: 3	vdepth: 1	vfrocost: 10	vparent: 1	visited: False
vid: 4	vdepth: 1	vfrocost: 10	vparent: 1	visited: False
vid: 5	vdepth: 0	vfrocost: inf	vparent: -1	visited: False
vid: 6	vdepth: 2	vfrocost: 11	vparent: 3	visited: False
vid: 7	vdepth: 2	vfrocost: 11	vparent: 3	visited: False
vid: 8	vdepth: 0	vfrocost: inf	vparent: -1	visited: False
vid: 9	vdepth: 3	vfrocost: 12	vparent: 6	visited: False
vid: 10	vdepth: 0	vfrocost: inf	vparent: -1	visited: False


最后得TEDDA树  图结构如下： 
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

EDDA计算得到的传输代价为：31； EDDA树边的连接关系如下：（）为同一层节点
0-1 - (2 - 3 - 4)
0-7 - (6)
0-9
1-2
1-3
1-4
7-6

---------------- 执行：EDDIP --------------
EDDIP方法的总价值为：23; 找到的传输方法为：
0 - (1- 6)
1 - (0- 2- 3- 4)
2 - (1)
3 - (1)
4 - (1)
6 - (0- 7- 9)
7 - (6)
9 - (6)

----------------- 执行：MMR --------------
当前进入 MMR2 的子算法 先生成Tc树 
 原始图信息为：
G={
0	 adjacency:[(1, 9), (2, 9), (3, 9), (4, 9), (5, 9), (6, 9), (7, 9), (8, 9), (9, 9), (10, 9)]	 connecty: 10	 visited: False	 isR: False	 isT_ms: False,
1	 adjacency:[(2, 1), (3, 1), (4, 1), (0, 9)]	 connecty: 4	 visited: False	 isR: True	 isT_ms: False,
2	 adjacency:[(1, 1), (5, 1), (4, 1), (0, 9)]	 connecty: 4	 visited: False	 isR: True	 isT_ms: False,
3	 adjacency:[(1, 1), (6, 1), (7, 1), (0, 9)]	 connecty: 4	 visited: False	 isR: True	 isT_ms: False,
4	 adjacency:[(1, 1), (2, 1), (10, 1), (0, 9)]	 connecty: 4	 visited: False	 isR: True	 isT_ms: False,
5	 adjacency:[(2, 1), (8, 1), (0, 9)]	 connecty: 3	 visited: False	 isR: False	 isT_ms: False,
6	 adjacency:[(3, 1), (7, 1), (9, 1), (0, 9)]	 connecty: 4	 visited: False	 isR: True	 isT_ms: False,
7	 adjacency:[(3, 1), (6, 1), (8, 1), (0, 9)]	 connecty: 4	 visited: False	 isR: True	 isT_ms: False,
8	 adjacency:[(5, 1), (7, 1), (9, 1), (0, 9)]	 connecty: 4	 visited: False	 isR: False	 isT_ms: False,
9	 adjacency:[(6, 1), (8, 1), (10, 1), (0, 9)]	 connecty: 4	 visited: False	 isR: True	 isT_ms: False,
10	 adjacency:[(4, 1), (9, 1), (0, 9)]	 connecty: 3	 visited: False	 isR: False	 isT_ms: False,
}

当前SMT树（Tc）的总代价为： 15 ； Tc树 的图结构如下：
G={
0	 adjacency:[(1, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
1	 adjacency:[(0, 9), (2, 1), (3, 1), (4, 1)]	 connecty: 4	 visited: False	 isR: False	 isT_ms: False,
2	 adjacency:[(1, 1)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
3	 adjacency:[(1, 1), (6, 1), (7, 1)]	 connecty: 3	 visited: False	 isR: False	 isT_ms: False,
4	 adjacency:[(1, 1)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
6	 adjacency:[(3, 1), (9, 1)]	 connecty: 2	 visited: False	 isR: False	 isT_ms: False,
7	 adjacency:[(3, 1)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
9	 adjacency:[(6, 1)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
}

当前的mmr 访问过的节点信息是：
vid: 0	vdepth: -1	vfrocost: 0	vparent: -1	visited: False
vid: 1	vdepth: 0	vfrocost: 9	vparent: 0	visited: False
vid: 2	vdepth: 1	vfrocost: 10	vparent: 1	visited: False
vid: 3	vdepth: 1	vfrocost: 10	vparent: 1	visited: False
vid: 4	vdepth: 1	vfrocost: 10	vparent: 1	visited: False
vid: 6	vdepth: 2	vfrocost: 11	vparent: 3	visited: False
vid: 7	vdepth: 2	vfrocost: 11	vparent: 3	visited: False
vid: 9	vdepth: 3	vfrocost: 12	vparent: 6	visited: False

MMR 算法结束，调整后的Tc数 的图结构如下：
G={
0	 adjacency:[(1, 9), (6, 9), (7, 9), (9, 9)]	 connecty: 4	 visited: False	 isR: False	 isT_ms: False,
1	 adjacency:[(0, 9), (2, 1), (3, 1), (4, 1)]	 connecty: 4	 visited: False	 isR: False	 isT_ms: False,
2	 adjacency:[(1, 1)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
3	 adjacency:[(1, 1)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
4	 adjacency:[(1, 1)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
6	 adjacency:[(0, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
7	 adjacency:[(0, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
9	 adjacency:[(0, 9)]	 connecty: 1	 visited: False	 isR: False	 isT_ms: False,
}

当前的mmr 访问过的节点信息是：
vid: 0	vdepth: -1	vfrocost: 0	vparent: -1	visited: False
vid: 1	vdepth: 0	vfrocost: 9	vparent: 0	visited: False
vid: 2	vdepth: 1	vfrocost: 10	vparent: 1	visited: False
vid: 3	vdepth: 1	vfrocost: 10	vparent: 1	visited: False
vid: 4	vdepth: 1	vfrocost: 10	vparent: 1	visited: False
vid: 6	vdepth: 0	vfrocost: 1	vparent: 0	visited: True
vid: 7	vdepth: 0	vfrocost: 1	vparent: 0	visited: True
vid: 9	vdepth: 0	vfrocost: 1	vparent: 0	visited: True
MMR计算得到的输出代价为：39 ；调整后的Tc树的连接关系为：（）为同一层节点
0-1 - (2 - 3 - 4)
0-6
0-7
0-9
1-2
1-3
1-4

"""