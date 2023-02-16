#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import math
import numba
import time
from sklearn.model_selection import StratifiedKFold
# 用来生成随机整数，用于k折中每次划分已知类
import random


# 根据随机设定的第一个已知类下标index以及已知类个数sub_size，选取已知类，并将数据集x、y中对应已知类的样本提取出来。
@numba.jit
# (nopython=True)
def subsample_classes_origin(x, y, index, sub_size, seed):
    # print("出错了 所以看看subsample里的参数index是什么：",index)
    # print("出错了 所以看看subsample里的参数np.unique(y)是什么：", np.unique(y))
    # print("出错了 所以看看subsample里的参数y.shape是什么：", y.shape)
    labs = np.unique(y[y != index])  # 取出和achor不一样类别的类别
    np.random.seed(seed)
    sub_labs = np.random.choice(labs, size=(sub_size - 1,), replace=False)  # 随机选取sub_size(想要子采样的类别数)-1个类别
    keep_classes = np.append(sub_labs, index)  # 子采样的类别： 随机选取的sub_size-1个 + achor类别

    idx_keep = np.array([np.any(keep_classes == u) for u in y])  # 选取满足子采样类别的样本
    y_new = y[idx_keep]
    x_new = x[idx_keep]

    ref = np.unique(keep_classes)

    return x_new, y_new, ref

# 根据随机设定的第一个已知类下标index以及已知类个数sub_size，选取已知类，并将数据集x、y中对应已知类及未知类的样本分别提取出来。
# Proprocessing data, i.e. subsampling data to split as known / unknown
@numba.jit
# (nopython=True)
def choose_known_classes(x, y, index, sub_size, seed):
    # print("出错了 所以看看subsample里的参数index是什么：",index)
    # print("出错了 所以看看subsample里的参数np.unique(y)是什么：", np.unique(y))
    # print("出错了 所以看看subsample里的参数y.shape是什么：", y.shape)
    labs = np.unique(y[y != index])  # 取出和achor不一样类别的类别
    np.random.seed(seed)
    sub_labs = np.random.choice(labs, size=(sub_size - 1,), replace=False)  # 随机选取sub_size(想要子采样的类别数)-1个类别
    keep_classes = np.append(sub_labs, index)  # 子采样的类别： 随机选取的sub_size-1个 + achor类别

    # 满足为已知类的
    idx_keep = np.array([np.any(keep_classes == u) for u in y])  # 选取满足子采样类别的样本
    y_new = y[idx_keep]
    x_new = x[idx_keep]

    # 不满足为已知类的，即为未知类的
    idx_others = np.array([np.all(keep_classes != u) for u in y])
    y_others = y[idx_others]
    x_others = x[idx_others]

    ref = np.unique(keep_classes)

    return x_new, y_new, ref, x_others, y_others

# 根据已知类ref，对样本重新打标签：已知类样本的标签更新为“原标签+1”（使得标签从1开始），未知类样本的标签更新为“-99999”（方便使用libsvm算法进行开放集识别）
@numba.jit
# (nopython=True)
def relabel_smaple(label, ref):
    temp = np.where(ref == label)[0]
    if temp.shape[0] == 0:  # 不属于ref中的类别
        return -99999
    else:  # 属于ref中的某个类别
        return temp[0] + 1

# 调用_temp_fun_to_relabel方法，根据已知类ref，对样本重新打标签
@numba.jit
# (nopython=True)
def relabel(labs, ref):
    new_labs = np.array([ relabel_smaple(u, ref) for u in labs])

    return new_labs


# 计算数据集X和数据集Y中每对样本间的欧氏距离
@numba.jit(nopython=True, parallel=True, nogil=True)
def pairwise_distance_matrix(X, Y):
    Nx = X.shape[0]
    Ny = Y.shape[0]
    P = X.shape[1]

    D = np.empty((Nx, Ny), dtype=numba.float32)

    for i in numba.prange(Nx):
        for j in numba.prange(Ny):
            d = 0.0
            for k in numba.prange(P):
                tmp = X[i, k] - Y[j, k]  # 计算 X第i个样本 和 Y第j个样本 在第k个特征上的距离
                d += tmp * tmp  # 平方 并 累加
            D[i, j] = np.sqrt(d)  # 求平方
    return D


###############################################################################################################################

# 1. OSNN
# 计算某一批次的样本的distance ratio
@numba.jit(nopython=True, parallel=True)
def _compute_OSNN_on_batch(dMat, y_train):

    N_batch = dMat.shape[0]                 # 验证/测试样本数
    y_preds = np.zeros(shape=(N_batch,))
    distance_ratio = np.zeros(shape=(N_batch,))

    for i in numba.prange(N_batch):         # 对于每一个验证/测试样本
        idx_sort = dMat[i, :].argsort()     # 根据Euclidean distance排序，返回排序后的下标
        dSorted = dMat[i][idx_sort]         # 返回排序后的Euclidean distance
        labs_sorted = y_train[idx_sort]     # 返回排序后的Euclidean distance对应的标签
        # neighbor_classes = labs_sorted[:2]
        y_pred = labs_sorted[0]             # 最近样本的标签
        nume = dSorted[0]                   # 距离最近的样本对应的Euclidean distance
        if y_pred == labs_sorted[1]:        # 如果 距离最近的样本的标签 和 距离第2近的样本的标签 一样
            keep_idx = (labs_sorted != y_pred)  # 取 和距离最近的样本标签不一样的 距离近的样本 的下标
            nc_dists = dSorted[keep_idx]    # 对应取它的Euclidean distance
            denom = nc_dists[0]
        else:
            denom = dSorted[1]

        ratio = nume / (denom + 1e-09)  # Added for numerical stability, i.e. if denom too close to zero

        y_preds[i] = y_pred
        distance_ratio[i] = ratio

    return distance_ratio, y_preds


# 按批次大小batch_size计算数据集X_test相对于数据集X_train的distance ratio（先计算X_test和X_train每对样本间的欧式距离，再根据欧式距离矩阵计算X_test的distance ratio）
@numba.jit(nopython=True, parallel=True)
def predict_OSNN(X_train, y_train, X_test, batch_size):

    N_test = X_test.shape[0]  # 验证样本个数
    distance_ratio = np.zeros(shape=(N_test,))  # 填充0
    y_preds = np.zeros(shape=(N_test,))

    # 分批次
    batch_size = np.min(np.array([batch_size, N_test]))
    num_steps = numba.int32(N_test / batch_size)  # 将样本分割为numba.int32(N_test / batch_size)批，每次大小为batch_size

    for step in numba.prange(num_steps):

        lb = step * batch_size          # 第step批次的起始行
        ub = (step + 1) * batch_size    # 第step批次的最末行

        if (ub + batch_size) > N_test:
            ub = N_test                 # 考虑最后一个批次不足batch_size的情况，将其合并在上一批次中

        X_batch = X_test[lb:ub, :]      # 获取第step批次的数据
        idx_batch = np.arange(lb, ub)

        dMat = pairwise_distance_matrix(X_batch,
                                        X_train)
        # 计算验证/测试样本和拟合样本的pairwise disatnces matrix，大小为（batch_size,X_train_size）

        # print("OSNN里的X_batch × X_train：", X_batch.shape, X_train.shape)
        # print("OSNN里的dMat.shape:", dMat.shape)

        distance_ratio[idx_batch], y_preds[idx_batch] = _compute_OSNN_on_batch(dMat, y_train)
        # 根据pairwise distances matrix以及拟合样本的标签计算验证/测试样本的distance ratio以及预测的标签

    return distance_ratio, y_preds

# 根据样本的distance ratio与阈值的相对大小，进行类别识别
@numba.jit(nopython=True)
def classify_OSNN(y_pred, distance_ratio, threshold):
    # y_pred是样本对应的最近样本的标签
    idx = (distance_ratio > threshold)      # 取distance ratio大于阈值的样本下标
    preds = y_pred.copy()
    preds[idx] = -99999                     # 将其标签更新为未知类标签（-99999）

    return preds

# 计算阈值范围内每一个阈值选择t的识别NA
@numba.jit
# (nopython=True)
def get_optimal_threshold_OSNN(X_train, y_train, X_test, y_test, T, batch_size, lamda_r=0.5):
    dists, ypreds = predict_OSNN(X_train, y_train, X_test, batch_size)  # 得到样本对应的distance ratio以及prediction

    tl = T.shape[0]
    final_NA = np.zeros(shape=(tl,))
    NA = np.zeros(shape=(tl,))
    AKS_array = np.zeros(shape=(tl,))
    AUS_array = np.zeros(shape=(tl,))
    for t in range(tl):
        thrs = T[t]
        newpreds = classify_OSNN(distance_ratio=dists, y_pred=ypreds,
                                 threshold=thrs)  # 若dists大于阈值thrs，则为unknown；否则还是ypreds

        num_cls = len(np.unique(y_test))
        # 1, 2, ..., num_cls-1 为 known class; -99999为 unknown class

        A = []  # 某类识别正确的样本数
        A_num = []  # 属于某类的样本数
        for i in range(1, num_cls):  # 计算对应类的识别accuracy
            idx = (y_test == i)
            y_p = newpreds[idx]
            A.append(np.sum(y_p == i))
            A_num.append(np.sum(y_test == i))  # np.count_nonzero(labels == i) 用于计算第i类的样本数

        idx = (y_test == -99999)
        y_p = newpreds[idx]
        A.append(np.sum(y_p == -99999))
        A_num.append(np.sum(y_test == -99999))

        A = np.array(A)
        A_num = np.array(A_num)
        AKS = np.sum(A[:num_cls - 1]) / (np.sum(A_num[:num_cls - 1]) + 1e-08)
        AUS = A[num_cls - 1] / (A_num[num_cls - 1] + 1e-08)
        AKS_array[t] = AKS
        AUS_array[t] = AUS
        NA[t] = lamda_r * AKS + (1 - lamda_r) * AUS
        final_NA[t]=NA[t]

    return final_NA, AKS_array, AUS_array

# 利用k折交叉验证，每次用get_optimal_threshold_OSNN方法求OSNN方法的最优阈值（根据最大平均NA）
@numba.jit
# (nopython=True)
def threshold_estimator_OSNN(X_train, y_train, T, batch_size,
                             random_state=1, lamda_r=0.5, n_splits=10):

    uniClasses = np.unique(y_train)  # 训练样本类别
    nClasses = uniClasses.shape[0]  # 训练样本类别数
    n_keep = numba.int32(nClasses / 2) + 1

    # 改成用k折交叉验证，来求阈值
    t_shape = T.shape[0]
    NA_array = np.zeros(shape=(n_splits, t_shape))
    AKS = np.zeros(shape=(n_splits, t_shape))
    AUS = np.zeros(shape=(n_splits, t_shape))

    skf = StratifiedKFold(n_splits=n_splits, random_state=21, shuffle=True)
    # 利用k折交叉验证，将数据集拆分成训练集和测试集
    n_index = 0
    for train_ind, val_ind in skf.split(X_train, y_train):
        train_x = X_train[train_ind]
        train_y = y_train[train_ind]
        val_x = X_train[val_ind]
        val_y = y_train[val_ind]

        # 划分已知类和未知类
        sed = random.randint(0, 20)
        np.random.seed(sed)
        anchor = np.random.choice(uniClasses, 1, False)  # 从所有类别中随机选择一个anchor

        train_vector, train_label, ref, _, _ = choose_known_classes(train_x, train_y, anchor, n_keep,
                                                                    seed=random_state * n_index + 7)
        # print(f'The following classes were selected as the known classes for validation: {list(ref)}')

        train_label = relabel(labs=train_label, ref=ref)
        val_vector = val_x
        val_label = relabel(labs=val_y, ref=ref)

        NA_array[n_index], AKS[n_index], AUS[n_index] = get_optimal_threshold_OSNN(train_vector,train_label,
                                                                                     val_vector,val_label,
                                                                                     T,batch_size,lamda_r)
        n_index += 1

    # print("-----------------------NA_array:", NA_array)
    value = np.mean(NA_array, axis=0)  # 按列求平均
    # print("-----------------------value:", value)
    opt_t_index = np.where(value == np.max(value))
    opt_t = T[opt_t_index]
    value_AKS = np.mean(AKS, axis=0)  # 按列求平均
    value_AUS = np.mean(AUS, axis=0)  # 按列求平均
    print("OSNN找阈值时的最优NA：", value[opt_t_index])
    print("OSNN找阈值时的最优NA对应的AKS：", value_AKS[opt_t_index])
    print("OSNN找阈值时的最优NA对应的AUS：", value_AUS[opt_t_index])

    return opt_t

###############################################################################################################################

# 2. OSNN_CV
# 按批次大小batch_size计算数据集X_test相对于数据集X_train的distance ratio（先计算X_test和X_train每对样本间的欧式距离，再根据欧式距离矩阵计算X_test的distance ratio）
# 并根据最近样本标签和第二近样本标签的异同，进行开放集识别
@numba.jit(nopython=True, parallel=True)
def predict_OSNN_CV(X_train, y_train, X_test, batch_size):
    N_test = X_test.shape[0]  # 验证样本个数
    y_preds = np.zeros(shape=(N_test,))

    # print("predict_OSNN_CV里的np.unique(y_train).shape[0]：", np.unique(y_train).shape[0])
    # print("predict_OSNN_CV里的np.unique(y_train):", np.unique(y_train))

    # 分批次
    batch_size = np.min(np.array([batch_size, N_test]))
    num_steps = numba.int32(N_test / batch_size)  # 将样本分割为numba.int32(N_test / batch_size)批，每次大小为batch_size

    for step in numba.prange(num_steps):

        lb = step * batch_size  # 第step批次的起始行
        ub = (step + 1) * batch_size  # 第step批次的最末行

        if (ub + batch_size) > N_test:
            ub = N_test  # 考虑最后一个批次不足batch_size的情况，将其合并在上一批次中

        X_batch = X_test[lb:ub, :]  # 获取第step批次的数据
        idx_batch = np.arange(lb, ub)

        dMat = pairwise_distance_matrix(X_batch,
                                        X_train)  # Compute the pairwise distances as matrix: (batch_size, X_train_size)
        # 计算验证/测试样本和拟合样本的pairwise disatnces matrix，大小为（batch_size,X_train_size）

        # print("OSNN_CV里的dMat.shape:", dMat.shape)

        y_preds[idx_batch] = classify_OSNN_CV(dMat, y_train)
        # 根据pairwise distances matrix以及拟合样本的标签计算验证/测试样本的预测标签

    # 或许可以在这里relabel修改classify_OSNN_CV中的unknown类标签
    # pred_num = np.unique(y_preds).shape[0]  # 首先获取到标签数
    # print("np.unique(y_preds).shape[0]  # 首先获取到标签数:",np.unique(y_preds).shape[0])
    # y_train_num = y_train.shape[0]          # 获取到y_train数
    # print("y_train.shape[0]  # 获取到y_train数:",y_train.shape[0])
    # y_preds[y_preds == y_train_num] = pred_num-1
    # print("修改后不一样的y_preds：",np.unique(y_preds))

    return y_preds

# 根据最近样本标签和第二近样本标签的异同，进行开放集识别
@numba.jit(nopython=True)
def classify_OSNN_CV(dMat, y_train):
    '''
    INTERNAL NUMBA FUNCTION USED IN OpenGPD MODULE
    ----------------------------------------------

    If both nearest neighbors have the same label, this label is assigned to the test sample. Otherwise, the observation is classified as unknown.
    '''

    N_batch = dMat.shape[0]  # 验证/测试样本数
    y_preds = np.zeros(shape=(N_batch,))

    # print("classify_OSNN_CV里的np.unique(y_train).shape[0]：",np.unique(y_train).shape[0])
    # print("classify_OSNN_CV里的np.unique(y_train):",np.unique(y_train))

    for i in numba.prange(N_batch):         # 对于每一个验证/测试样本
        idx_sort = dMat[i, :].argsort()     # 根据Euclidean distance排序，返回排序后的下标
        labs_sorted = y_train[idx_sort]     # 返回排序后的Euclidean distance对应的标签

        if (labs_sorted[0] == labs_sorted[1]):  # 如果最近样本标签和第二近样本标签相同
            y_preds[i] = labs_sorted[0]         # 则识别为最近样本标签
        else:
            y_preds[i] = -99999                 # 否则识别为未知类

    return y_preds

###############################################################################################################################

# 3. NN
# 按批次大小batch_size计算数据集X_test相对于数据集X_train的distance ratio（先计算X_test和X_train每对样本间的欧式距离，再根据欧式距离矩阵计算X_test的distance ratio）
# 并根据最近样本标签，进行开放集识别
@numba.jit(nopython=True, parallel=True)
def predict_NN(X_train, y_train, X_test, batch_size):

    N_test = X_test.shape[0]  # 验证样本个数
    y_preds = np.zeros(shape=(N_test,))

    # print("predict_NN里的np.unique(y_train).shape[0]：", np.unique(y_train).shape[0])
    # print("predict_NN里的np.unique(y_train):", np.unique(y_train))

    # 分批次
    batch_size = np.min(np.array([batch_size, N_test]))
    num_steps = numba.int32(N_test / batch_size)  # 将样本分割为numba.int32(N_test / batch_size)批，每次大小为batch_size

    for step in numba.prange(num_steps):

        lb = step * batch_size  # 第step批次的起始行
        ub = (step + 1) * batch_size  # 第step批次的最末行

        if (ub + batch_size) > N_test:
            ub = N_test  # 考虑最后一个批次不足batch_size的情况，将其合并在上一批次中

        X_batch = X_test[lb:ub, :]  # 获取第step批次的数据
        idx_batch = np.arange(lb, ub)

        dMat = pairwise_distance_matrix(X_batch,
                                        X_train)  # Compute the pairwise distances as matrix: (batch_size, X_train_size)
        # 计算验证/测试样本和拟合样本的pairwise disatnces matrix，大小为（batch_size,X_train_size）

        # print("NN里的dMat.shape:", dMat.shape)

        y_preds[idx_batch] = classify_NN(dMat, y_train)
        # 根据pairwise distances matrix以及拟合样本的标签计算验证/测试样本的预测标签

    return y_preds

# 将样本识别为最近样本对应的标签
@numba.jit(nopython=True)
def classify_NN(dMat, y_train):

    N_batch = dMat.shape[0]  # 验证/测试样本数
    y_preds = np.zeros(shape=(N_batch,))

    # print("classify_NN里的np.unique(y_train).shape[0]：",np.unique(y_train).shape[0])
    # print("classify_NN里的np.unique(y_train):",np.unique(y_train))

    for i in numba.prange(N_batch):  # 对于每一个验证/测试样本
        idx_sort = dMat[i, :].argsort()  # 根据Euclidean distance排序，返回排序后的下标
        labs_sorted = y_train[idx_sort]  # 返回排序后的Euclidean distance对应的标签

        y_preds[i] = labs_sorted[0]

    return y_preds

###############################################################################################################################

# 4. TNN
# 按批次大小batch_size计算数据集X_test相对于数据集X_train的distance ratio（先计算X_test和X_train每对样本间的欧式距离，再根据欧式距离矩阵计算X_test的distance ratio）
# 并根据最近样本标签，暂时进行开放集识别
@numba.jit(nopython=True, parallel=True)
def predict_TNN(X_train, y_train, X_test, batch_size):

    N_test = X_test.shape[0]  # 验证样本个数
    distance = np.zeros(shape=(N_test,))  # 填充0
    y_preds = np.zeros(shape=(N_test,))

    # 分批次
    batch_size = np.min(np.array([batch_size, N_test]))
    num_steps = numba.int32(N_test / batch_size)  # 将样本分割为numba.int32(N_test / batch_size)批，每次大小为batch_size

    for step in numba.prange(num_steps):

        lb = step * batch_size  # 第step批次的起始行
        ub = (step + 1) * batch_size  # 第step批次的最末行

        if (ub + batch_size) > N_test:
            ub = N_test  # 考虑最后一个批次不足batch_size的情况，将其合并在上一批次中

        X_batch = X_test[lb:ub, :]  # 获取第step批次的数据
        idx_batch = np.arange(lb, ub)

        dMat = pairwise_distance_matrix(X_batch,
                                        X_train)  # Compute the pairwise distances as matrix: (batch_size, X_train_size)
        # 计算验证/测试样本和拟合样本的pairwise disatnces matrix，大小为（batch_size,X_train_size）

        # print("TNN里的dMat.shape:", dMat.shape)
        # print("dMat:",dMat)

        distance[idx_batch], y_preds[idx_batch] = classify_TNN_first(dMat, y_train)
        # 根据pairwise distances matrix以及拟合样本的标签计算验证/测试样本的distance ratio以及预测的标签

    return distance, y_preds

# 暂时将样本识别为最近样本对应的标签
@numba.jit(nopython=True)
def classify_TNN_first(dMat, y_train):
    '''

    The observation is classified as the label of the nearest neighbor first taking no account of the threshold.

    '''

    N_batch = dMat.shape[0]  # 验证/测试样本数
    distance = np.zeros(shape=(N_batch,))
    y_preds = np.zeros(shape=(N_batch,))

    for i in numba.prange(N_batch):         # 对于每一个验证/测试样本
        idx_sort = dMat[i, :].argsort()     # 根据Euclidean distance从小到大排序，返回排序后的下标
        dSorted = dMat[i][idx_sort]         # 返回排序后的Euclidean distance
        labs_sorted = y_train[idx_sort]     # 返回排序后的Euclidean distance对应的标签

        distance[i] = dSorted[0]
        y_preds[i] = labs_sorted[0]

    return distance, y_preds

# 根据样本的暂时标签、distance ratio及阈值，进行开放集识别
@numba.jit(nopython=True)
def classify_TNN(y_pred, distance_ratio, threshold):
    idx = (distance_ratio > threshold)        # 取distance ratio大于阈值的样本下标
    preds = y_pred.copy()
    preds[idx] = -99999                 # 将其标签更新为未知类标签（-99999）

    return preds

# 计算阈值范围内每一个阈值选择t的识别NA
@numba.jit
# (nopython=True)
def get_optimal_threshold_TNN(X_train, y_train, X_test, y_test, T, batch_size, lamda_r=0.5):

    dists, ypreds = predict_TNN(X_train, y_train, X_test, batch_size)  # 得到样本对应的distance ratio以及prediction

    tl = T.shape[0]
    final_NA = np.zeros(shape=(tl,))
    NA = np.zeros(shape=(tl,))
    AKS_array = np.zeros(shape=(tl,))
    AUS_array = np.zeros(shape=(tl,))
    for t in range(tl):
        thrs = T[t]
        newpreds = classify_TNN(distance_ratio=dists, y_pred=ypreds,
                                 threshold=thrs)  # 若dists大于阈值thrs，则为unknown；否则还是ypreds

        num_cls = len(np.unique(y_test))
        # 1, 2, ..., num_cls-1 为 known class; -99999 为 unknown class

        A = []  # 某类识别正确的样本数
        A_num = []  # 属于某类的样本数
        for i in range(1, num_cls):  # 计算对应类的识别accuracy
            idx = (y_test == i)
            y_p = newpreds[idx]
            A.append(np.sum(y_p == i))
            A_num.append(np.sum(y_test == i))  # np.count_nonzero(labels == i) 用于计算第i类的样本数

        idx = (y_test == -99999)
        y_p = newpreds[idx]
        A.append(np.sum(y_p == -99999))
        A_num.append(np.sum(y_test == -99999))

        A = np.array(A)
        A_num = np.array(A_num)
        AKS = np.sum(A[:num_cls - 1]) / (np.sum(A_num[:num_cls - 1]) + 1e-08)
        AUS = A[num_cls - 1] / (A_num[num_cls - 1] + 1e-08)
        AKS_array[t] = AKS
        AUS_array[t] = AUS
        NA[t] = lamda_r * AKS + (1 - lamda_r) * AUS
        final_NA[t] = NA[t]

    return final_NA, AKS_array, AUS_array

# 利用k折交叉验证，每次用get_optimal_threshold_OSNN方法求OSNN方法的最优阈值（根据最大平均NA）
@numba.jit
# (nopython=True)
def threshold_estimator_TNN(X_train, y_train, T, batch_size,
                            random_state=1, lamda_r=0.5, n_splits=10):

    uniClasses = np.unique(y_train)  # 训练样本类别
    nClasses = uniClasses.shape[0]  # 训练样本类别数
    n_keep = numba.int32(nClasses / 2) + 1

    # 改成用k折交叉验证，来求阈值
    t_shape = T.shape[0]
    NA_array = np.zeros(shape=(n_splits, t_shape))
    AKS = np.zeros(shape=(n_splits, t_shape))
    AUS = np.zeros(shape=(n_splits, t_shape))

    skf = StratifiedKFold(n_splits=n_splits, random_state=21, shuffle=True)
    # 利用k折交叉验证，将数据集拆分成训练集和测试集
    n_index = 0
    for train_ind, val_ind in skf.split(X_train, y_train):
        train_x = X_train[train_ind]
        train_y = y_train[train_ind]
        val_x = X_train[val_ind]
        val_y = y_train[val_ind]

        # 划分已知类和未知类
        sed = random.randint(0, 20)
        np.random.seed(sed)
        anchor = np.random.choice(uniClasses, 1, False)  # 从所有类别中随机选择一个anchor

        train_vector, train_label, ref, _, _ = choose_known_classes(train_x, train_y, anchor, n_keep,
                                                                  seed=random_state * n_index + 7)
        # print(f'The following classes were selected as the known classes for validation: {list(ref)}')

        train_label = relabel(labs=train_label, ref=ref)
        val_vector = val_x
        val_label = relabel(labs=val_y, ref=ref)

        NA_array[n_index], AKS[n_index], AUS[n_index] = get_optimal_threshold_TNN(train_vector,
                                                                                   train_label,
                                                                                   val_vector,
                                                                                   val_label,
                                                                                   T,
                                                                                   batch_size,
                                                                                   lamda_r)
        n_index += 1

    # print("-----------------------NA_array:", NA_array)
    value = np.mean(NA_array, axis=0)  # 按列求平均
    # print("-----------------------value:", value)
    opt_t_index = np.where(value == np.max(value))
    opt_t = T[opt_t_index]
    value_AKS = np.mean(AKS, axis=0)  # 按列求平均
    value_AUS = np.mean(AUS, axis=0)  # 按列求平均
    print("TNN找阈值时的最优NA：", value[opt_t_index])
    print("TNN找阈值时的最优NA对应的AKS：", value_AKS[opt_t_index])
    print("TNN找阈值时的最优NA对应的AUS：", value_AUS[opt_t_index])

    return opt_t

###############################################################################################################################

# 5. OSNN_GPD
# 计算每一批次的训练集的distance ratio
@numba.jit
# (nopython=True, parallel=True)
def _train_train_distance_ratio_on_batch(X_batch, y_batch, X_out_of_batch, y_out_of_batch):

    N_batch = y_batch.shape[0]
    y_preds_batch = np.zeros(shape=(N_batch,))
    distance_ratio_batch = np.zeros(shape=(N_batch,))
    dMat_batch = pairwise_distance_matrix(X_batch, X_batch)
    dMat_out_batch = pairwise_distance_matrix(X_batch, X_out_of_batch)

    for i in numba.prange(N_batch):
        # X_batch和X_batch
        dists = dMat_batch[i]
        dists = np.delete(dists, i)  # 删掉自己和自己的distance
        y_batch_new = np.delete(y_batch, i)  # 同时在标签列表中删掉自己的标签
        idx_batch_sort = dists.argsort()  # 对X_batch和X_batch的distance从小到大排序
        dSorted_batch = dists[idx_batch_sort]
        labs_sorted_batch = y_batch_new[idx_batch_sort]  # 对应的标签

        # X_batch和X_out_of_batch
        dists_out = dMat_out_batch[i]
        idx_out_batch_sort = dists_out.argsort()
        dSorted_out_batch = dists_out[idx_out_batch_sort]
        labs_sorted_out_batch = y_out_of_batch[idx_out_batch_sort]

        # 进行拼接
        new_dists = np.concatenate((dSorted_batch, dSorted_out_batch))
        newlabs = np.concatenate((labs_sorted_batch, labs_sorted_out_batch))

        idx_sort = new_dists.argsort()  # 对拼接后的进行排序
        dSorted = new_dists[idx_sort]
        labs_sorted = newlabs[idx_sort]
        neighbor_classes = labs_sorted[:1]  # 取最近的样本对应的标签
        y_pred = neighbor_classes

        nume = dSorted[:1]

        keep_idx = (labs_sorted != y_pred)  # 取不属于“y_pred”的样本
        nc_dists = dSorted[keep_idx]
        denom = np.sum(nc_dists[:1])  # 计算最近的不属于“y_pred”的样本与当前样本的距离的和
        ratio = nume / (denom + 1e-09)  # 计算distance ratio
        # Added for numerical stability, i.e. if denom too close to zero

        y_preds_batch[i] = y_pred  # 暂时预测为与“最近的样本对应的标签”同一类
        distance_ratio_batch[i] = ratio

    return distance_ratio_batch, y_preds_batch


# 计算训练集相对于训练集的distance ratio（用的留一法）
@numba.jit
# (nopython=True, parallel=True)
def Train_train_distance_ratio(X_train, y_train, batch_size):

    N_train = X_train.shape[0]
    distance_ratio = np.zeros(shape=(N_train,))
    y_preds = np.zeros(shape=(N_train,))

    # set up slices to make batches
    batch_size = np.min(np.array([batch_size, N_train]))
    num_steps = numba.int32(N_train / batch_size)

    # 分批次计算X_train的distance ratio
    for step in numba.prange(num_steps):

        lb = step * batch_size  # Get index of starting row of batch
        ub = (step + 1) * batch_size  # Get index of ending row of batch

        if (ub + batch_size) > N_train:
            ub = N_train  # If the next batch does not contain batch_size points, merge with current batch

        idx_batch = np.repeat(numba.b1(0), N_train)
        idx_batch[lb:ub] = numba.b1(1)
        idx_out_batch = np.repeat(numba.b1(1), N_train)
        idx_out_batch[lb:ub] = numba.b1(0)
        X_batch = X_train[idx_batch]
        y_batch = y_train[idx_batch]
        X_out_batch = X_train[idx_out_batch]
        y_out_batch = y_train[idx_out_batch]

        distance_ratio[lb:ub], y_preds[lb:ub], = _train_train_distance_ratio_on_batch(X_batch,
                                                                                      y_batch,
                                                                                      X_out_batch,
                                                                                      y_out_batch)
    return distance_ratio, y_preds

# 计算每一批次的测试/验证集相对于训练集的distance ratio
@numba.jit
# (nopython=True, parallel=True)
def _test_train_distance_ratio_on_batch(dMat, y_train):

    N_batch = dMat.shape[0]
    y_preds = np.zeros(shape=(N_batch,))
    distance_ratio = np.zeros(shape=(N_batch,))
    for i in numba.prange(N_batch):
        idx_sort = dMat[i, :].argsort()  # 对distance从小到大排序
        dSorted = dMat[i][idx_sort]  # 排序后对应的距离
        labs_sorted = y_train[idx_sort]  # 排序后对应的标签
        neighbor_classes = labs_sorted[:1]  # 取最近的训练样本对应的标签
        y_pred = neighbor_classes

        nume = dSorted[:1]  # 计算最近的训练样本与当前样本距离

        keep_idx = (labs_sorted != y_pred)  # 取所有不属于“y_pred”的样本
        nc_dists = dSorted[keep_idx]  # 对它们的距离从小到大排序
        denom = np.sum(nc_dists[:1])  # 计算最近的不属于“y_pred”的训练样本与当前样本距离
        ratio = nume / (denom + 1e-09)  # 计算distance ratio
        # Added for numerical stability, i.e. if denom too close to zero

        y_preds[i] = y_pred  # 暂时预测为”最近样本的标签“同一类
        distance_ratio[i] = ratio

    return distance_ratio, y_preds


# 计算测试/验证集相对于训练集的distance ratio
@numba.jit
# (nopython=True, parallel=True)
def Test_train_distance_ratio(X_train, y_train, X_test, batch_size):

    N_test = X_test.shape[0]
    distance_ratio = np.zeros(shape=(N_test,))
    y_preds = np.zeros(shape=(N_test,))

    # 分批次
    batch_size = np.min(np.array([batch_size, N_test]))
    num_steps = numba.int32(N_test / batch_size)  # 将样本分割为numba.int32(N_test / batch_size)批，每次大小为batch_size

    for step in numba.prange(num_steps):

        lb = step * batch_size          # 第step批次的起始行
        ub = (step + 1) * batch_size    # 第step批次的最末行

        if (ub + batch_size) > N_test:
            ub = N_test                 # 考虑最后一个批次不足batch_size的情况，将其合并在上一批次中

        X_batch = X_test[lb:ub, :]      # 获取第step批次的数据
        idx_batch = np.arange(lb, ub)

        dMat = pairwise_distance_matrix(X_batch,
                                        X_train)  # Compute the pairwise distances as matrix: (batch_size, X_train_size)

        distance_ratio[idx_batch], y_preds[idx_batch] = _test_train_distance_ratio_on_batch(dMat, y_train)
    return distance_ratio, y_preds


# 计算gpd参数的方法
# 计算Wτ,γ
@numba.jit(nopython=True)
def _FUN_tau(exc, gamma, tau):
    n = exc.shape[0]
    frac_vec = exc / (tau ** 2 - exc * tau + 1e-10)
    OUT = -1 * n / tau - (1 / gamma + 1) * np.sum(frac_vec)
    return OUT


# 计算∂Wτ,γ/∂τ
@numba.jit(nopython=True)
def _FUN_tau_deriv(exc, gamma, tau):
    n = exc.shape[0]
    frac_vec = (exc * (2 * tau - exc)) / ((tau ** 2 - exc * tau + 1e-10) ** 2)
    OUT = n / (tau ** 2) + (1 / gamma + 1) * np.sum(frac_vec)
    return OUT


# Estimate parameters of GPD using maximum likelihood with (gamma, tau) parametarization.
# 根据τ0，计算γ0，以及迭代iterations计算τ、γ
@numba.jit(nopython=True)
def compute_gpd_parameter_by_initial_value(distance, threshold_percentile, iterations, tau_init):

    # Note since some distances are labeled -999 we remove them before fitting
    distance = distance[distance >= 0]
    if 0 <= threshold_percentile <= 100:
        thres = np.percentile(a=distance, q=threshold_percentile)  # a: 用来计算分位数的对象     q:介于0-100的float，用来计算分位数的参数
        diffs = distance - thres
        exc = diffs[diffs > 0]  # 取distance ratio比阈值thres要大的

        n = exc.shape[0]
        prior = n / distance.shape[0]  # distance ratio比阈值thres要大的 占 所有distance ratio的比例

        tau = tau_init  # τ的初始值
        gamma = 1 / n * np.sum(np.log(1 - exc / tau))  # γ的初始值

        for i in range(iterations):  # 迭代iterations次
            nume = _FUN_tau(exc=exc, gamma=gamma, tau=tau)  # 计算Wτ,γ
            denom = _FUN_tau_deriv(exc=exc, gamma=gamma, tau=tau)  # 计算∂Wτ,γ/∂τ
            tau -= nume / denom

            log_in = 1 - exc / tau
            idx = np.where(log_in <= 0)
            log_in[idx] = 1e-09
            gamma = 1 / n * np.sum(np.log(log_in))

        return gamma, tau, prior, thres

# 辅助在寻找最优阈值T时，比较性能的方法
# 计算Q_hat(j/(ne+1))
@numba.jit(nopython=True)
def _quantileFunction(u, gamma, tau):
    Nu = u.shape[0]
    out = np.zeros(Nu)
    for i in range(Nu):
        out[i] = tau * (1 - (1 - u[i]) ** (-gamma))
    return out


# 识别方法
# 计算p(R>r(x*))=p(R>t)*(1-sj/τ)^(-1/γ)
@numba.jit(nopython=True)
def predict_gpd_pvalue(distance, gamma, tau, prior, threshold_numerical):
    diffs = distance - threshold_numerical

    probs = np.ones(shape=distance.shape)
    for i in range(probs.shape[0]):
        exc = diffs[i]
        if exc >= tau:
            probs[i] = 0
        elif exc >= 0:
            probs[i] = prior * (1 - exc / tau) ** (-1 / gamma)

    return probs


# 如果p(R>r(x*))≥α，则识别为之前预测的“最近的样本中对应的标签“；否则，识别为未知类。
@numba.jit(nopython=True)
def classify_OSNN_GPD(y_pred, probs, alpha):
    preds = y_pred.copy()
    idx = np.array([numba.b1(probs[i] < alpha) for i in range(preds.shape[0])])
    preds[idx] = -99999

    return preds


# 对t_percentrile_range对应的每一个阈值t，基于验证数据集X_val，做出开集识别预测结果，计算E(t)和C(t)，进而计算O(t|λ)
@numba.jit
# (nopython=True, parallel=True)
def parameter_estimator_OSNN_GPD_fold(t_percentile_range, alpha, penalty, X_train, y_train, X_val, y_val, tau_initial,
                             iterations, batch_size, lamda_r=0.5):
    # penalty就是论文中的λ
    num_k = 1
    num_t = t_percentile_range.shape[0]
    output_arr = np.zeros(shape=(num_k, num_t))

    dists_tr, prds_tr = Train_train_distance_ratio(X_train, y_train,
                                             batch_size)  # 计算X_train相对于X_train的distance ratio(用了留一法）
    dists_te, prds_before = Test_train_distance_ratio(X_train, y_train, X_val,
                                                   batch_size)  # 计算X_val相对于X_train的distance ratio
    err_before_osr = np.mean(prds_before != y_val)  # 将待测样本识别为“最近的样本对应的标签”的识别错误

    for j in range(num_t):
        # 根据选定的阈值，计算GPD的参数（用训练集）
        t = t_percentile_range[j]
        gamma, tau, prior, threshold = compute_gpd_parameter_by_initial_value(dists_tr, t, iterations, tau_initial)  # 迭代计算GPD的参数，返回(γ,τ,p(R>t),t)
        # 在测试集上看其性能
        orderStats = np.sort(dists_te - threshold)
        orderStats = orderStats[orderStats > 0]  # { sj | j=1, 2, ... , ne }
        probs = np.arange(1, orderStats.shape[0] + 1) / (orderStats.shape[0] + 1)  # j/(ne+1)
        quants = _quantileFunction(u=probs, gamma=gamma, tau=tau)  # Q_hat(j/(ne+1))
        corr = np.corrcoef(quants, orderStats)[1, 0]  # 皮尔逊相关系数矩阵

        osr_probs = predict_gpd_pvalue(dists_te, gamma, tau, prior, threshold)  # 计算p(R>r(x*))
        # prior: probability of exceeding threshold as returned by compute_gpd_parameter_by_initial_value. 其实就相当于p(R>t)
        prds_after = classify_OSNN_GPD(prds_before, osr_probs,alpha)  # 如果p(R>r(x*))≥α，则识别为之前预测的“最近样本的标签“；否则，识别为未知类。
        err_after_osr = np.mean(prds_after != y_val)

        # 根据识别的结果计算NA，进而比较参数的性能
        num_cls = len(np.unique(y_val))
        #  1, 2, ..., num_cls-1 为 known class; -99999为 unknown class

        A = []  # 某类识别正确的样本数
        A_num = []  # 属于某类的样本数
        for i in range(1, num_cls):  # 计算对应类的识别accuracy
            idx = (y_val == i)
            y_p = prds_after[idx]
            A.append(np.sum(y_p == i))
            A_num.append(np.sum(y_val == i))  # np.count_nonzero(labels == i) 用于计算第i类的样本数

        idx = (y_val == -99999)
        y_p = prds_after[idx]
        A.append(np.sum(y_p == -99999))
        A_num.append(np.sum(y_val == -99999))

        A = np.array(A)
        A_num = np.array(A_num)
        AKS = np.sum(A[:num_cls - 1]) / (np.sum(A_num[:num_cls - 1]) + 1e-08)
        AUS = A[num_cls - 1] / (A_num[num_cls - 1] + 1e-08)
        # AKS_array[t] = AKS
        # AUS_array[t] = AUS
        # NA[t] = lamda_r * AKS + (1 - lamda_r) * AUS
        # final_NA[t] = NA[t]

        output_arr[0, j] = (1 - penalty) * corr - penalty * (err_after_osr - err_before_osr)
        # output_arr[i, j] = penalty * (lamda_r * AKS + (1 - lamda_r) * AUS) + (1 - penalty) * corr

    return output_arr


# n_folds交叉验证：每次划分对应的训练集和验证集，调用上面的parameter_estimator_OSNN_GPD_fold方法，计算O(t|λ)，取最大的对应的t为optimal t；
# 再根据选取的t，计算GPD的参数τ和γ
@numba.jit
# (nopython=True)
def parameter_estimator_OSNN_GPD(X_train, y_train, t_percentile_range, alpha, penalty, tau_initial, iterations,
             batch_size, random_state, lamda_r=0.5, n_splits=10):
    num_k = 1
    num_t = t_percentile_range.shape[0]
    cv_obj = np.zeros(shape=(num_k, num_t))
    train_nsamples = y_train.shape[0]
    train_idx = np.arange(0, train_nsamples)
    np.random.seed(random_state)

    # 改成用k折交叉验证，来求阈值
    skf = StratifiedKFold(n_splits=n_splits, random_state=21, shuffle=True)
    # 利用k折交叉验证，将数据集拆分成训练集和测试集
    for train_ind, val_ind in skf.split(X_train, y_train):
        train_x = X_train[train_ind]
        train_y = y_train[train_ind]
        val_x = X_train[val_ind]
        val_y = y_train[val_ind]

        cv_obj += parameter_estimator_OSNN_GPD_fold(t_percentile_range,
                                                     alpha,
                                                     penalty,
                                                     train_x,
                                                     train_y,
                                                     val_x,
                                                     val_y,
                                                     tau_initial,
                                                     iterations,
                                                     batch_size,
                                                     lamda_r)
    row, col = np.where(cv_obj == np.max(cv_obj))
    opt_t_perc = t_percentile_range[col][0]  # 找到optimal t
    training_distances, _ = Train_train_distance_ratio(X_train, y_train,
                                                        batch_size)  # 基于求出的optimal t, 计算训练集相对于训练集的distance ratio（用的留一法）
    gamma, tau, prior, threshold_numerical = compute_gpd_parameter_by_initial_value(training_distances, opt_t_perc,iterations,tau_initial)
    # 根据τ0，计算γ0，以及迭代iterations计算τ、γ
    return gamma, tau, prior, threshold_numerical, opt_t_perc
