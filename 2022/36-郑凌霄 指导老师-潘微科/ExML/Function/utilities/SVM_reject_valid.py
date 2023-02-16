import numpy as np
import math
from Function.utilities.Reject import *

def SVM_reject_valid(y_tr,X_tr,conf_th_set,acc_min,hidden_class_id,opts,fold):
    # This function performs cross validation for selecting the rejection threshold
    # for the first layer of ExML to ensure desired accuracy on known classes

    # parameters setting

    conf_th_set = np.sort(conf_th_set)
    conf_min = conf_th_set[0]
    conf_max = conf_th_set[0]
    min_com = 1
    acc = 1000
    iteration = 0

    # assign index for cross validation
    fold_index = np.ones(np.shape(y_tr)[0]) * fold
    fold_num = np.floor(np.shape(y_tr)[0] / fold)
    for i in range(1,fold):
        fold_index[(i - 1) * fold_num: i * fold_num] = i

    fold_index = fold_index[np.randperm(np.shape(y_tr)[0])]

    # identifying the minimal thershold for desired accuracy via binary search

    while (abs(conf_max - conf_min) > 0.0101 and (abs(acc - acc_min) > 0.01 or math.isnan(acc))):
        conf_th = np.floor((conf_max + conf_min) / 2 * 1000 + 0.0001) / 1000
        acc_table = np.zeros((fold, 1))
        reject_rate = 0
        iteration = iteration + 1

        # evalute the accuracy on known classes by cross validation
        for iter_fold in range(1, fold+1):
            index_train_fold = fold_index != iter_fold
            index_valid_fold = fold_index == iter_fold

            y_tr_temp = y_tr[index_train_fold]
            X_tr_temp = X_tr[index_train_fold][:]
            y_te_temp = y_tr[index_valid_fold]
            X_te_temp = X_tr[index_valid_fold][:]
            M = pdist2(X_tr_temp, X_tr_temp)
            opts['kernel_para'] = np.median(M[:])

            # training rejection model with current threshold：
            [model_reject, labelSet] = rejectsvmtrain(y_tr_temp, X_tr_temp, 1 - conf_th, 1, 1, opts)
            [predict_new] = rejectsvmpredict(y_te_temp, X_te_temp, model_reject, labelSet, hidden_class_id)

            # record the accuracy on known classes
            index_high = (predict_new!=hidden_class_id)
            acc_temp = sum(y_te_temp(index_high) == predict_new(index_high)) / np.shape(y_te_temp[index_high])[0]
            reject_rate_temp = sum(predict_new == hidden_class_id) / max(np.shape(predict_new))
            acc_table[iter_fold] = acc_temp;
            reject_rate = reject_rate + reject_rate_temp

        acc = np.mean(acc_table)
        if math.isnan(acc):
            conf_max = conf_th
            continue

        reject_rate = reject_rate / fold
        comparator = reject_rate
        if (comparator < min_com and acc > acc_min and not math.isnan(comparator)):
            min_com = comparator
            conf_max = conf_th
        else:
            conf_min = conf_th

    theta_org = 1 - conf_th

    return theta_org
