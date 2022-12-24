# coding=utf-8
import sys
import pandas as pd
import numpy as np

#打印“用ground truth文件计算度量”，sys。Argv[1]， "和生成的输出"，sys.argv[2]
gt = pd.read_csv(sys.argv[1], header=None).as_matrix()
out = pd.read_csv(sys.argv[2], header=None).as_matrix()
thresh = float(sys.argv[3])

out[out <= thresh] = 0
# 将数组' out '中所有大于' thresh '的值设置为1。
out[out > thresh] = 1

index_arr = np.array(map(lambda x: \
        zip(range(gt.shape[1]), [x]*gt.shape[1]), range(gt.shape[0]))
        )

assert gt.shape == out.shape and gt.shape == index_arr.shape[:2]
TP = float(((gt+out) == 2).sum()) # Both 1
TN = float(((gt+out) == 0).sum()) # Both 0
FN = float(((gt-out) == 1).sum()) # Gt = 1, Out = 0 (False negative)
FP = float(((gt-out) == -1).sum()) # Gt = 0, Out = 1 (False positive)
print index_arr[(gt-out == 1)]
print "------------"
# 打印数组' index_arr '的下标，其中' out '和' gt '的差值为1。
print index_arr[(out-gt == 1)]

prec =  TP/(TP+FP)
recall =  TP/(TP+FN)
specificity = TN/(TN+FP)
#print "Precision:", prec
#print "Recall:", recall
#print "F1 Score:", 2*prec*recall/(prec+recall)
print prec, recall
