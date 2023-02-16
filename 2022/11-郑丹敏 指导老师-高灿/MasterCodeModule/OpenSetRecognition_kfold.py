#!/usr/bin/env python
# coding: utf-8

import numpy as np
import numba
import math

from UFuncs_osnn_kfold import *

# 构造开集环境
class PreprocessOSR:

    def __init__(self, X_vector, y_label, random_state):
        self.X_vector = X_vector
        self.y_label = y_label
        self.reference = None
        self.seed = random_state

    # 根据随机设定的第一个已知类下标index以及已知类个数sub_size，选取已知类
    def make_osr_data(self, index, sub_size):

        x_vector = self.X_vector
        y_label = self.y_label
        x_vector, y_label, ref, _, _ = choose_known_classes(x_vector, y_label, index, sub_size, self.seed)

        y_label = relabel(labs=y_label, ref=ref)

        self.X_vector = x_vector
        self.y_label = y_label
        self.reference = ref

        return self.X_vector, self.y_label

    # 根据选定的已知类，对样本重新打标签
    def relabel_classes(self, y_test):

        ref = self.reference
        y_new = relabel(labs=y_test, ref=ref)
        return y_new

# 评价指标
class PerformanceMeasures:

    def __init__(self):
        pass

    # 识别准确度
    def accuracy(self, labels, predictions):
        return np.mean(labels == predictions)

    #  the accuracy on known samples
    def AKS(self, labels, predictions):

        num_cls = len(np.unique(labels))
        # print("AKS里的np.unique(labels)：",np.unique(labels))
        # print("AKS里的np.unique(predictions)：", np.unique(predictions))
        # 1, 2, ..., num_cls-1 为 known class; -99999为 unknown class

        A = []  # 某类已知类中识别正确的个数
        A_num = []  # 某类已知类的个数
        MIS_array = []  # 将某类已知类错误识别为其他类
        FU_array = []  # 将已知类错误识别为未知类
        for i in range(1 , num_cls):  # 计算已知类的识别accuracy
            idx = (labels == i)
            y_p = predictions[idx]
            # A.append(np.sum(y_p == i)/np.count_nonzero(labels == i))    # np.count_nonzero(labels == i) 用于计算第i类的样本数
            A.append(np.sum(y_p == i))
            A_num.append(np.sum(labels == i))
            MIS_array.append(np.sum(y_p != i) - np.sum(y_p == -99999))
            FU_array.append(np.sum(y_p == -99999))

        A = np.array(A)
        A_num = np.array(A_num)
        MIS_array = np.array(MIS_array)
        FU_array = np.array(FU_array)
        # AKS = np.sum(A[:-1])/(num_cls-1)

        # print("计算AKS里的A[:-1]:",A[:-1])
        # print("计算AKS里的A[:num_cls-1]:",A[:num_cls-1])

        # AKS = np.sum(A[:-1])/np.sum(A_num[:-1])
        # MIS = np.sum(MIS_array[:-1])/np.sum(A_num[:-1])
        # FU = np.sum(FU_array[:-1]) / np.sum(A_num[:-1])

        AKS = np.sum(A[:num_cls - 1]) / (np.sum(A_num[:num_cls - 1]) + 1e-08)
        MIS = np.sum(MIS_array[:num_cls - 1]) / (np.sum(A_num[:num_cls - 1]) + 1e-08)
        FU = np.sum(FU_array[:num_cls - 1]) / (np.sum(A_num[:num_cls - 1]) + 1e-08)

        return AKS, MIS, FU

    #  the accuracy on unknown samples
    def AUS(self, labels, predictions):
        A = []
        idx = (labels == -99999)
        y_p = predictions[idx]
        A.append(np.sum(y_p == -99999) / (np.sum(labels == -99999) + 1e-08))
        # print("AUS计算:",np.sum(y_p == num_cls - 1) / np.count_nonzero(labels == num_cls - 1))

        A = np.array(A)
        AUS = A[0]

        return AUS

    # # The normalized accuracy (NA) takes into account both the accuracy on known samples (AKS) and the accuracy on unknown samples (AUS).
    # NA = lamda_r * AKS + (1 - lamda_r) * AUS
    def NA(self, labels, predictions, lamda_r=0.5):
        AKS, _, _ = self.AKS(labels, predictions)
        AUS = self.AUS(labels, predictions)
        NA = lamda_r * AKS + (1 - lamda_r) * AUS

        return NA

    # micro-averaging open-set f-measure
    def micro_F1(self, labels, predictions):
        num_cls = len(np.unique(labels))
        TP = []
        FP = []
        FN = []

        for i in range(1 , num_cls):
            idx = (labels == i)
            y_p = predictions[idx]

            TP.append(np.sum(y_p == i))  # 正确识别为第i类的
            FP.append(np.sum(predictions[labels != i] == i))  # 不是第i类但错误识别为第i类的
            FN.append(np.sum(y_p != i))  # 是第i类但错误识别为其他类的

        TP = np.array(TP)
        FP = np.array(FP)
        FN = np.array(FN)

        # PR = np.sum(TP[:-1])/np.sum(TP[:-1]+FP[:-1] + 1e-08)
        # RE = np.sum(TP[:-1])/np.sum(TP[:-1]+FN[:-1] + 1e-08)

        PR = np.sum(TP[:num_cls - 1]) / np.sum(TP[:num_cls - 1] + FP[:num_cls - 1] + 1e-08)
        RE = np.sum(TP[:num_cls - 1]) / np.sum(TP[:num_cls - 1] + FN[:num_cls - 1] + 1e-08)

        micro_F1_score = 2 * (PR * RE) / (PR + RE + 1e-08)

        return micro_F1_score

    # macro-averaging open-set f-measure
    def macro_F1(self, labels, predictions):
        num_cls = len(np.unique(labels))  # 1到num_cls为已知类
        TP = []
        FP = []
        FN = []

        for i in range(1, num_cls):
            idx = (labels == i)
            y_p = predictions[idx]

            TP.append(np.sum(y_p == i))
            FP.append(np.sum(predictions[labels != i] == i))
            FN.append(np.sum(y_p != i))

        TP = np.array(TP)
        FP = np.array(FP)
        FN = np.array(FN)

        # PR = np.sum(TP[:-1]/(TP[:-1]+FP[:-1] + 1e-08))/(num_cls-1)
        # RE = np.sum(TP[:-1]/(TP[:-1]+FN[:-1] + 1e-08))/(num_cls-1)

        PR = np.sum(TP[:num_cls - 1] / (TP[:num_cls - 1] + FP[:num_cls - 1] + 1e-08)) / (num_cls - 1)
        RE = np.sum(TP[:num_cls - 1] / (TP[:num_cls - 1] + FN[:num_cls - 1] + 1e-08)) / (num_cls - 1)

        macro_F1_score = 2 * (PR * RE) / (PR + RE + 1e-08)

        return macro_F1_score

# OSNN方法
class OSNN:
    """
    OSNN方法的特点在于不是直接对某一最相似类的相似性得分设置阈值，而是对两个最相似类的相似性得分比率设置阈值。分别计算待测样本到两个不同的最近邻u和t的欧式距离d(s,t)及d(s,u)，得到距离比R=d(s,t)/d(s,u)，如果R小于阈值，则s被分为和t相同的标签，否则就会被划分为未知类。
    为了得到OSNN方法中的阈值，通过在训练阶段模拟开放集环境，即选择训练集中一部分可用训练类为“未知类”，并基于给定的阈值取值范围及在该开放集环境中识别的最优性能估计阈值。
    """

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.threshold_fitted = False

    # 估计参数（阈值）
    def fit_threshold(self, threshold_range, batch_size, random_state, lamda_r=0.5, n_splits=10):

        opt_t = threshold_estimator_OSNN(X_train=self.X_train,
                                         y_train=self.y_train,
                                         T=threshold_range,
                                         batch_size=batch_size,
                                         random_state=random_state,
                                         lamda_r=lamda_r,
                                         n_splits=n_splits)

        self.threshold = np.mean(opt_t)
        print("OSNN选定的阈值T为:", self.threshold)
        self.threshold_fitted = True
        return None

    # 计算测试集相对于训练集的distance ratio，并基于选定的阈值，将待测样本识别为对应的已知类或者未知类
    def predict(self, X_test, batch_size):
        # 检查是否已经估计阈值了
        if not self.threshold_fitted:
            print("Model must first be fitted.")
            return None
        self.distance_ratio, self.pred = predict_OSNN(self.X_train, self.y_train, X_test, batch_size)
        self.predictions = classify_OSNN(self.pred, self.distance_ratio, self.threshold)
        return self.predictions

# OSNN_CV方法
class OSNN_CV:
    """
    OSNN_CV方法的思想是在预测阶段选出待测样本的两个最近邻，如果两个具有相同的标签，则把对应的标签分给该待测样本，否则该待测样本会被划分为未知类。
    """

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # 计算测试集与测试集间每对样本的欧式距离，选取待测样本的两个最近邻样本，根据这两个样本标签的异同，将待测样本识别为对应的已知类或未知类。
    def predict(self, X_test, batch_size):
        # print("OSNN_CV.predict里的np.unique(y_train).shape[0]：", np.unique(self.y_train).shape[0])
        self.predictions = predict_OSNN_CV(self.X_train, self.y_train, X_test, batch_size)

        return self.predictions

# NN方法
class NN:
    """
    NN方法用于将样本识别为最近邻样本对应的类别。NN方法只能用于闭集环境。
    """

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    # 计算测试集与测试集间每对样本的欧式距离，将待测样本识别为其最近邻样本对应的类别
    def predict(self, X_test, batch_size):
        # print("NN.predict里的np.unique(y_train).shape[0]：", np.unique(self.y_train).shape[0])
        self.predictions = predict_NN(self.X_train, self.y_train, X_test, batch_size)

        return self.predictions

# TNN方法
class TNN:
    """
    TNN（NN using Threshold）方法在NN的基础上设置了一个阈值，当待测样本与最近邻样本的距离大于该阈值时，则将该待测样本划分为未知类。
    """

    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.threshold_fitted = False

    # 寻找TNN的最优阈值
    # 在T的网格搜索中，我们尝试从0到√D的值和100个线性分离的值，其中D是数据集的特征数
    def fit_threshold(self, threshold_range, batch_size, random_state, lamda_r=0.5, n_splits=10):
        opt_t = threshold_estimator_TNN(self.X_train,
                                         self.y_train,
                                         threshold_range,
                                         batch_size,
                                         random_state,
                                         lamda_r,
                                         n_splits)

        self.threshold = np.mean(opt_t)
        print("TNN选定的阈值T为:", self.threshold)
        self.threshold_fitted = True

        return None

    # 计算测试集与测试集间每对样本的欧式距离，根据待测样本与其最近邻样本的距离与选定的阈值的相对大小，将待测样本识别为对应的已知类或未知类
    def predict(self, X_test, batch_size):
        # 检查是否已经估计阈值了
        if not self.threshold_fitted:
            print("Model must first be fitted.")
            return None
        # print("TNN.predict里的np.unique(y_train).shape[0]：", np.unique(self.y_train).shape[0])
        self.distance, self.pred = predict_TNN(self.X_train, self.y_train, X_test, batch_size)
        # print("看一下y_train:", self.y_train)
        # print("看一下X_test:", X_test)
        # print("看一下self.distance:", self.distance)
        # print("看一下self.pred:", self.pred)
        # print("看一下self.threshold:", self.threshold)
        self.predictions = classify_TNN(self.pred, self.distance, self.threshold)
        # print("看一下self.predictions:", self.predictions)

        return self.predictions

# OSNN_GPD相关的辅助方法
class Auxiliary:
    # 计算测试集相对于训练集的distance ratio，以及基于最近的训练样本标签的预测
    def Test_Train_DR(self, X_test, batch_size):
        if not self.parameter_fitted:
            print("Parameter must first be fitted.")
            return None
        X_train = self.X_train
        y_train = self.y_train
        self.distance_ratio_test, self.predictions_knn = Test_train_distance_ratio(X_train, y_train, X_test,
                                                                                batch_size)
        return None

    # 计算训练集相对于训练集的distance ratio，并根据最近的样本出现次数最多的标签给出预测；并将训练集距离比计算标志置为true。
    def Train_Train_DR(self, batch_size):
        if not self.parameter_fitted:
            print("Parameter must first be fitted.")
            return None
        X_train = self.X_train
        y_train = self.y_train
        self.distance_ratio_train, self.knn_predictions_training_data = Train_train_distance_ratio(X_train, y_train,
                                                                                             batch_size)
        self.train_train_distance_ratio = True
        return None

    # 计算迭代求GPD的参数τ和γ
    def compute_gpd_parameter(self, threshold_percentile, tau_initial=1, iterations=1000):
        if not self.train_train_distance_ratio:
            print('First compute the distance ratio on the training data using the Train_Train_DR method.')
            return None
        distance = self.distance_ratio_train
        self.gamma, self.tau, self.prior, self.threshold = compute_gpd_parameter_by_initial_value(distance, threshold_percentile, iterations,tau_initial)
        return None

    # 计算p(R>r(x*))=p(R>t)*(1-sj/τ)^(-1/γ)
    def pvalue(self, threshold=None):
        if np.any(self.distance_ratio_test == None):
            print("First compute distance ratio of validation set to estimate parameters.")
            return None
        if self.gamma == None:
            print("First fit the GPD to the distance ratios.")
            return None
        if not threshold == None:
            self.threshold = threshold
        distance = self.distance_ratio_test
        gamma = self.gamma
        tau = self.tau
        prior = self.prior
        self.gpd_pvalue = predict_gpd_pvalue(distance, gamma, tau, prior, self.threshold)
        return None

    # 如果p(R>r(x*))≥α，则识别为之前预测的“最近的样本对应的标签“；否则，识别为未知类。
    def classification_KOSNN_GPD(self, alpha):
        if np.any(self.distance_ratio_test == None):
            print("First compute distance ratio of test set.")
            return None
        if np.any(self.gpd_pvalue == None):
            print("First compute GPD p-value.")
            return None

        self.predictions_kosnn_gpd = classify_OSNN_GPD(self.predictions_knn, self.gpd_pvalue, alpha)
        return None

class OSNN_GPD(Auxiliary):
    """
    OSNN_GPD方法在OSNN的基础上使用广义Pareto分布
    """
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.parameter_fitted = False           # 参数（k和阈值T）是否已经被估计了
        self.predictions_knn = None             # 基于最近k个近邻样本出现次数最多的标签识别待测样本
        self.train_train_distance_ratio = False  # 训练集相对于训练集的distance ratio是否已经计算了
        self.distance_ratio_train = None        # 训练集相对于训练集的distance ratio
        self.distance_ratio_set = None          # 测试集相对于训练集的distance ratio
        self.threshold_percentile = None        # gpd中的阈值T（百分比）
        self.threshold = None                   # gpd中的阈值T（根据训练集相对于训练集的distance ratio以及百分比得到）
        self.gamma = None                       # gpd参数γ
        self.tau = None                         # gpd参数τ
        self.prior = None                       # distance ratio大于阈值的概率，即P(R>t)；用于计算分布R中大于某个测试样本的distance ratio的概率P(R>r(x*))
        self.gpd_pvalue = None                  # 分布R中大于某个测试样本的distance ratio的概率，即p(R>r(x*))
        self.predictions_kosnn_gpd = None       # 识别的结果
        super().__init__()

    # 求optimal t
    def fit_parameter(self, t_range, alpha, penalty, batch_size, tau_initial=1,
                 iterations=1000, random_state=9, lamda_r=0.5, n_splits = 10):
        X_train = self.X_train
        y_train = self.y_train
        gamma, tau, prior, threshold_numerical, opt_t_perc = parameter_estimator_OSNN_GPD(X_train, y_train,
                                                                                                 t_range, alpha, penalty,
                                                                                                 tau_initial, iterations, batch_size,random_state,lamda_r, n_splits)
        self.gamma = gamma
        self.tau = tau
        self.prior = prior
        self.threshold = threshold_numerical
        self.threshold_percentile = opt_t_perc
        self.parameter_fitted = True
        return None

    def predict(self, X_test, alpha, batch_size, threshold=None):
        if not self.parameter_fitted:
            print("Parameter must first be fitted.")
            return None
        super().Test_Train_DR(X_test, batch_size)   # 计算待测样本相对于训练集的distance ratio，以及基于最近k个训练样本标签的预测
        super().pvalue(threshold)                   # 计算p(R>r(x*))=p(R>t)*(1-sj/τ)^(-1/γ)
        super().classification_KOSNN_GPD(alpha)     # 将p(R>r(x*))与α比较，实现开放集识别
        return self.predictions_kosnn_gpd, self.gpd_pvalue