import numpy as np
import random
import math
import matplotlib.pyplot as plt


# 读取数据
def read_data(data_path):
    data = []
    with open(data_path) as data_temp:
        while True:
            line = data_temp.readline()
            if not line:
                break
            data.append([float(i) for i in line.split()])
    return np.array(data)

# 目标函数
def goal_function(n, lambda_tra, gamma_tra, sg_a, data_x, diagonal_weight, trace):
    goal_number = 0
    for i in range(n):
        for j in range(n):
            two_norm_number = np.sum(pow(diagonal_weight * data_x[i][:].T - diagonal_weight * data_x[j][:].T, 2))
            goal_number += gamma_tra * (sg_a[i][j]**2) + two_norm_number * sg_a[i][j]
    goal_number += 2 * lambda_tra * trace
    return goal_number

def updata_sg_a(v):
    # 解决 min 1/2||x-v||^2 ，输入v，得到x
    ft = 1
    n = len(v)
    x = np.zeros(n)
    v0 = v - np.sum(v)/n + 1/n
    vmin = v0.min()
    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 10**(-4):
            v1 = lambda_m - v0
            posidx = np.zeros(n)
            for i in range(n):
                if v1[i] > 0:
                    posidx[i] = 1

            npos = np.sum(posidx)
            g = npos/n - 1
            f = 0
            for i in range(n):
                if posidx[i] > 0:
                    f += v1[i]
            f = f/n - lambda_m
            lambda_m = lambda_m - f/g
            ft = ft + 1
            if ft > 100:
                for i in range(n):
                    if v1[i] < 0:
                        x[i] = -v1[i]
                break
        for i in range(n):
            if v1[i] < 0:
                x[i] = -v1[i]
    else:
        x = v0

    return x

def main(data_x , cluster_c):

    '''

    在此SWCAN代码中，我复现了作者论文中函数递归推导实现的各个函数，包括目标函数以及其变化曲线，
    但可能局部细节未处理好，所以效果并没有达到作者文中体现的。

    '''

    # 参数个数
    n = data_x.shape[0]
    # k邻近点个数
    #k_number = int(n/cluster_c)
    k_number = 10
    # 初始化参数
    lambda_tra = k_number
    gamma_tra = 0.01

    # 特征个数
    d = data_x.shape[1]
    # 随机初始化权重向量
    vector_weight = np.random.rand(d, 1)
    # print(vector_weight)
    # 对角权重矩阵
    diagonal_weight = np.zeros((d, d))
    for i in range(d):
        diagonal_weight[i][i] = vector_weight[i]

    # 相似性图构建
    sg_a = np.zeros((n, n))
    for i in range(n):
        temp_distance = []
        for j in range(n):
            if j == i:  # 同一个点不考虑
                continue
            distance = math.sqrt(np.sum(pow(data_x[i] - data_x[j], 2)))
            sg_a[i][j] = distance
            temp_distance.append(distance)
        temp_distance.sort()
        temp_sum = sum(temp_distance[:k_number])
        kst_min_distance = temp_distance[k_number - 1]
        # 将距离不是k个最近邻近点的两点距离赋值为0
        for j in range(n):
            if sg_a[i][j] > kst_min_distance:
                sg_a[i][j] = 0
            else:
                sg_a[i][j] = sg_a[i][j]/temp_sum
    # sg_a = np.zeros((n, n))
    # for i in range(n):
    #     temp_sum = 0
    #     for j in range(n):
    #         if j == i:  # 同一个点不考虑
    #             continue
    #         distance = math.sqrt(np.sum(pow(data_x[i] - data_x[j], 2)))
    #         sg_a[i][j] = distance
    #         temp_sum += distance
    #     for j in range(n):
    #         sg_a[i][j] = sg_a[i][j]/temp_sum


    # 拉普拉斯矩阵L
    sg_a_d = np.zeros((n, n))
    for i in range(n):
        sg_a_d[i][i] = (np.sum(sg_a[i][:]) + np.sum(sg_a[:][i]))/2
    l_matrix = sg_a_d - (sg_a + sg_a.T)/2

    # 初始F
    eigenvalue, featurevector = np.linalg.eig(l_matrix)
    eigenvalue_c = eigenvalue[n-cluster_c:]
    trace = np.sum(eigenvalue_c)
    featurevector_c = featurevector[n - cluster_c:][:]
    featurevector_c = featurevector_c.T
    # print(featurevector_c)

    # 初始权重向量
    m = data_x.T@l_matrix@data_x  # 三个矩阵相乘
    sum_back_m = 0
    for i in range(d):
        sum_back_m += 1 / m[i][i]
    for i in range(d):
        vector_weight[i][0] = 1 / (m[i][i] * sum_back_m)

    # 记录用于画图
    goalNumber = []

    # 迭代目标函数出口值
    goal_differ_number = 0.001
    goal_function_number_pre = goal_function(n, lambda_tra, gamma_tra, sg_a, data_x, diagonal_weight, trace)
    print(goal_function_number_pre)
    # 迭代次数
    iter_number = 0
    # 迭代梯度下降
    while True:

        goalNumber.append(goal_function_number_pre)

        # 更新拉普拉斯矩阵L
        sg_a_d = np.zeros((n, n))
        for i in range(n):
            sg_a_d[i][i] = (np.sum(sg_a[i][:]) + np.sum(sg_a[:][i])) / 2
        l_matrix = sg_a_d - (sg_a + sg_a.T) / 2

        # 更新F
        eigenvalue, featurevector = np.linalg.eig(l_matrix)
        eigenvalue_c = eigenvalue[n - cluster_c:]
        trace = np.sum(eigenvalue_c)
        featurevector_c = featurevector[n - cluster_c:][:]
        featurevector_c = featurevector_c.T
        # print(eigenvalue_c)
        # print(featurevector_c)
        # print(trace)
        # print(featurevector_c)

        # 更新权重向量
        m = data_x.T @ l_matrix @ data_x  # 三个矩阵相乘
        sum_back_m = 0
        tiny_constant = 0.001   # 用于防止m太小，以至于1/m趋向无穷大
        for i in range(d):
            if m[i][i] < tiny_constant:
                m[i][i] = tiny_constant
            sum_back_m += 1 / m[i][i]
        for i in range(d):
            vector_weight[i][0] = 1 / (m[i][i] * sum_back_m)
        # print(vector_weight)
        # 对角权重矩阵
        for i in range(d):
            diagonal_weight[i][i] = vector_weight[i]

        # 更新相似图A
        r = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                two_norm_number = np.sum(pow(diagonal_weight * data_x[i].T - diagonal_weight * data_x[j].T, 2))
                f_two_norm_number = np.sum(pow(featurevector_c[i] - featurevector_c[j], 2))
                r[i][j] = two_norm_number + lambda_tra * f_two_norm_number
            v = -(1/2 * gamma_tra) * r[i]
            sg_a[i] = updata_sg_a(v)

        # 更新目标函数
        goal_function_number_que = goal_function(n, lambda_tra, gamma_tra, sg_a, data_x, diagonal_weight, trace)
        # print(goal_function_number_que)

        iter_number += 1
        # 迭代上限
        if iter_number > 30:
            break

        # 目标函数出口
        if abs(goal_function_number_pre - goal_function_number_que) < goal_differ_number:
            break


        goal_function_number_pre = goal_function_number_que



    # 迭代次数
    print('迭代次数为：', iter_number)

    # 输出得到的相似图A
    print(sg_a)

    # 画目标函数值变化曲线
    x = range(1, iter_number)
    y = goalNumber[1:]
    plt.scatter(x, y, s=4)
    plt.plot(x, y, color='r', linestyle='--')
    plt.xlabel('iteration')
    plt.ylabel('goalNumber')
    plt.show()



if __name__ == '__main__':
    data_x = read_data('data.txt')
    cluster_c = 3
    main(data_x, cluster_c)