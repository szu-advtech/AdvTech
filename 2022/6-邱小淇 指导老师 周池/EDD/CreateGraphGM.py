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
（1）初始化：给定N个节点和待添加的边数M
（2）随机连边：
	①随机选取一对没有边相连的不同的节点，并在这对节点之间添加一条边。
	②重复步骤①，直至在M对不同的节点对之间各添加了一条边。

"""
def GM(NETWORK_SIZE=10,COUNT_OF_EAGE=10):
    print('请输入ER网络的顶点个数N：'+str(NETWORK_SIZE))
    #NETWORK_SIZE = int(input())
    print('请输入连边数量M：'+str(COUNT_OF_EAGE))
    #COUNT_OF_EAGE = int(input())
    adjacentMatrix = np.zeros((NETWORK_SIZE, NETWORK_SIZE), dtype=int)  # 初始化邻接矩阵
    rvertex=[]
    random.seed(time.time())  # 'random.random()#生成[0,1)之间的随机数


    # 生成ER网络
    def generateRandomNetwork():
        count = 0

        while(count < COUNT_OF_EAGE):
            points = []
            while(len(points)<2):
                point = random.randrange(NETWORK_SIZE)

                flag = True
                for i in range(NETWORK_SIZE):
                    if(i == point):
                        continue
                    elif(adjacentMatrix[point][i] == 1):
                        flag = False
                    elif(adjacentMatrix[point][i] == 0):
                        flag=True
                        points.append(int(i))
                        points.append(int(point))

                        break

            print(str(points[0]+1) +' - ' + str(points[1]+1))
            if points[0] not in rvertex:
                rvertex.append(points[0])
            if points[1] not in rvertex:
                rvertex.append(points[1])
            if (adjacentMatrix[points[0]][points[1]] == 0) and (adjacentMatrix[points[0]][points[1]] == 0):
                adjacentMatrix[points[0]][points[1]] = adjacentMatrix[points[1]][points[0]] = 1
                count += 1
            else:
                continue


    # 用于绘制ER图
    def showGraph():
        G = nx.Graph()
        vnum=len(adjacentMatrix)

        G.add_nodes_from(range(1,vnum+1))
        for i in range(len(adjacentMatrix)):
            for j in range(len(adjacentMatrix)):
                if adjacentMatrix[i][j] == 1:  # 如果不加这句将生成完全图，ER网络的邻接矩阵将不其作用
                    G.add_edge(i+1, j+1)
        nx.draw(G,with_labels=True)
        plt.show()


    # 将ER网络写入文件中
    def writeRandomNetworkToFile():
        ARRS = []
        f = open('randomNetwork02.txt', 'w+')
        for i in range(NETWORK_SIZE):
            t = adjacentMatrix[i]
            ARRS.append(t)
            for j in range(NETWORK_SIZE):
                s = str(t[j])
                f.write(s)
                f.write(' ')
            f.write('\n')
        #写入r节点
        for i in rvertex:
            f.write(str(i+1))
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
    print('您所构造的G(N,M)ER网络如下：')
    #showGraph()

