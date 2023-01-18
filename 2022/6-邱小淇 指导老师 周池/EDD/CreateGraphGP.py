# -*- coding: utf-8 -*-
"""
reconstruction demo of the paper:《Cost-Effective App Data Distribution in Edge Computing》
year：2022
language：pythoy 3.7
ceated by:siki

"""
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import random
import time
"""
（1）初始化：给定N个节点以及连边概率p∈[0,1]
（2）随机连边：
	①随机选取一对没有边相连的不同的节点
	②生成一个随机数r∈[0,1]
	③如果r＜p，那么在这对节点之间添加一条边；否则就不添加
	④重复步骤①~③，直至所有节点对都被选择一次
"""
print('请输入ER网络的顶点个数：')
NETWORK_SIZE = int(input())
print('请输入图的密度d：')
d=float(input())
p= d*2/(NETWORK_SIZE-1)
PROBABILITY_OF_EAGE = float(p)
adjacentMatrix = np.zeros((NETWORK_SIZE, NETWORK_SIZE), dtype=int)  # 初始化邻接矩阵
random.seed(time.time())  # 'random.random()#生成[0,1)之间的随机数


# 生成ER网络
def generateRandomNetwork():
    count = 0
    probability = 0.0
    for i in range(NETWORK_SIZE):
        for j in range(i + 1, NETWORK_SIZE):
            probability = random.random()
            if probability < PROBABILITY_OF_EAGE:
                count = count + 1
                adjacentMatrix[i][j] = adjacentMatrix[j][i] = 1
    print('您所构造的ER网络边数为：' + str(count))
    print(adjacentMatrix)


# 用于绘制ER图2
def showGraph():
    G = nx.Graph()
    for i  in range(NETWORK_SIZE):
        G.add_node(i)
    for i in range(len(adjacentMatrix)):
        for j in range(len(adjacentMatrix)):
            if adjacentMatrix[i][j] == 1:  # 如果不加这句将生成完全图，ER网络的邻接矩阵将不其作用
                G.add_edge(i, j)
    nx.draw(G)
    plt.show()


# 将ER网络写入文件中
def writeRandomNetworkToFile():
    rvertex=[i for i in range(NETWORK_SIZE)]
    print(rvertex)
    random.shuffle(rvertex)
    ARRS = []
    f = open('randomNetwork01.txt', 'w+')
    for i in range(NETWORK_SIZE):
        t = adjacentMatrix[i]
        ARRS.append(t)
        for j in range(NETWORK_SIZE):
            s = str(t[j])
            f.write(s)
            f.write(' ')
        f.write('\n')
    # 写入r节点
    for i in rvertex:
        f.write(str(i + 1))
        f.write(' ')

    f.write('\n')
    f.close()




# 主程序开始
start = time.perf_counter()  # 用以程序计时开始位置
generateRandomNetwork()  # 生成ER随机网络
writeRandomNetworkToFile()  # 将随机网络写入randomNetwork01.txt文件中
#calculateDegreeDistribution()  # 计算此ER随机网络的度分布并将结果写入文件degreee01.txt文件中
finish = time.perf_counter()  # 程序计时结束
duration = finish - start
print('生成这个ER网络需要的时间为：' + str(duration) + 's')
print('您所构造的G(N,p)ER网络如下：')
showGraph()


