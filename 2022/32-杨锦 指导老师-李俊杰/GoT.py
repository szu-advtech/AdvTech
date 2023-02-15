import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
import copy
import heapq
from sklearn import metrics

data = np.loadtxt("data/1_Iris", delimiter=",")
n = np.shape(data)[0]
K = 8
CA_matrix = [[0 for i in range(n)] for i in range(n)]
DM_matrix = [[0 for i in range(n)] for i in range(n)]
RM_matrix = [[0 for i in range(n)] for i in range(n)]
density = [0 for i in range(n)]
point_numbers_in_a_cluster = [0 for i in range(K)]#每个簇中样本点的数量

def compute_distances_no_loops(A, B):
    '''计算两个矩阵的距离矩阵'''
    return cdist(A,B,metric='euclidean')


def plotFeature(data, labels_):
    '''显示簇集'''
    clusterNum=len(set(labels_))
    fig = plt.figure()
    scatterColors = ['orange', 'blue', 'green', 'yellow', 'red', 'purple', 'black', 'brown','#BC8F8F','#8B4513','#FFF5EE']
    ax = fig.add_subplot(111)
    for i in range(-1,clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels_==i)]
        ax.scatter(subCluster[:,0], subCluster[:,1], c=colorSytle, s=20)
    plt.show()

def floyd(matrix=list()):
    """弗洛伊德算法,构造RM矩阵需要用到此算法"""
    matrix = copy.deepcopy(matrix)

    for i in range(n):
        # i 表示中介顶点的下标
        for k in range(n):
            # k 表示出发点的下标
            for j in range(n):
                # j表示目标点的下标
                if k == j:
                    matrix[k][j] = 0
                    continue
                temp = matrix[k][i] + matrix[i][j]
                if matrix[k][j] > temp:
                    matrix[k][j] = temp
    return matrix


# 需要聚类的数据data
# K 簇的数量
# tol 聚类的容差，即ΔJ
# 聚类迭代都最大次数N
def K_means(data,K,tol,N):
    '''基础聚类算法'''

    #centerId是初始中心向量的索引坐标
    centerId = random.sample(range(0, n), K)
    # 获得初始中心向量,k个

    centerPoints = data[centerId]

    # 计算data到centerPoints的距离矩阵
    # dist[i][:],是i个点到三个中心点的距离
    dist = compute_distances_no_loops(data, centerPoints)

    # axis=1寻找每一行中最小值都索引
    # getA()是将mat转为ndarray
    # squeeze()是将label压缩成一个列表
    labels = np.argmin(dist, axis=1).squeeze()

    # 初始化old J
    oldVar = -0.0001
    # data - centerPoint[labels]，获得每个向量与中心向量之差
    # np.sqrt(np.sum(np.power(data - centerPoint[labels], 2)，获得每个向量与中心向量距离
    # 计算new J
    newVar = np.sum(np.sqrt(np.sum(np.power(data - centerPoints[labels], 2), axis=1)))
    # 迭代次数
    count=0
    # 当ΔJ大于容差且循环次数小于迭代次数，一直迭代。负责结束聚类
    # abs(newVar - oldVar) >= tol:
    while count<N and abs(newVar - oldVar) > tol:
        oldVar = newVar
        for i in range(K):
            # 重新计算每一个类别都中心向量
            centerPoints[i] = np.mean(data[np.where(labels == i)], 0)
        # 重新计算距离矩阵
        dist = compute_distances_no_loops(data, centerPoints)
        # 重新分类
        labels = np.argmin(dist, axis=1).squeeze()
        # 重新计算new J
        newVar = np.sum(np.sqrt(np.sum(np.power(data - centerPoints[labels], 2), axis=1)))
        # 迭代次数加1
        count+=1
    # 返回类别标识，中心坐标
    return labels,centerPoints



#构造CA矩阵,同时得到每个点的 density 和 representative capacity
#times为进行聚类的次数
times = 20
for time in range(times):
    labels, centerPoints = K_means(data,K,0.01,100)
    for i in range(n):
        point_numbers_in_a_cluster[labels[i]] += 1
        for j in range(n):
            if labels[i] == labels[j]:
                CA_matrix[i][j] += 1

for i in range(n):
    for j in range(n):
        CA_matrix[i][j] /= times
#至此，CA矩阵构造完成

#计算每个点的初始density
for i in range(n):
    density[i] = point_numbers_in_a_cluster[labels[i]]/(n*times)

#根据公式（6）更新density
for i in range(n):
    for j in range(i,n):
        if density[i] == density[j]:
            diff = random.uniform(0,0.1)
            density[i] -= diff
            break
#density计算完成

#构造DM矩阵
for i in range(n):
    for j in range(n):
        DM_matrix[i][j] = 1-CA_matrix[i][j]
#至此，DM矩阵构造完成

#利用弗洛伊德算法构造RM矩阵
RM_matrix = floyd(DM_matrix)

#计算每个点的representative capacity
capacity = [RM_matrix[i][0] for i in range(n)]
for i in range(n):
    for j in range(n):
        if density[j] > density[i] and RM_matrix[i][j] < capacity[i]:
            capacity[i] = RM_matrix[i][j]
#capicity计算完成

#计算每个点的tendency
tendency = [density[i] * capacity[i] for i in range(n)]

#找到tendency最大的K个点
max_tendency_ID = heapq.nlargest(K,range(n),key = lambda i:tendency[i])
max_tendency = data[max_tendency_ID]

#对K个中心点重新设置标签
for i in range(K):
    labels[max_tendency_ID[i]] = i

#每个样本点与tendency最大的K个点之间的距离矩阵
distance = compute_distances_no_loops(data, max_tendency)


#计算每个点的margin，并将margin>th的点加入到对应的簇，其他点舍弃，th的值可自行设计
th = 0.05
margin = [0 for i in range(n)]
for i in range(n):
    if i not in max_tendency_ID:

        smallest_two_distance = heapq.nsmallest(2,range(K),key = lambda j:distance[i][j])
        margin[i] = abs(distance[i][smallest_two_distance[0]] - distance[i][smallest_two_distance[1]])

        if (margin[i] > th):
            labels[i] = labels[max_tendency_ID[smallest_two_distance[0]]]
        else:
            labels[i] = K

#从数据文件中得到真实的tag
true_tag = []
with open("data/1_Iris") as file1:
    for line in file1:
        true_tag.append(int(line[0]))

#利用ARI和NMI评估集成结果
ARI_score = metrics.adjusted_rand_score(true_tag, labels)
NMI_score = metrics.normalized_mutual_info_score(true_tag, labels)
print('ARI_score = ' + str(ARI_score))
print('NMI_score = ' + str(NMI_score))

plotFeature(data, labels)