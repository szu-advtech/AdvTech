from libsvm.svm import *
from libsvm.svmutil import *
from sklearn import svm
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
    #import scikitplot as skplt


from sklearn import datasets


train_num =460
test_num =660
#data = pd.read_csv('xx.csv')
train_data = data.values[0:train_num,1:]
train_label = data.values[0:train_num,0]
# test_data = data.values[train_num:test_num ,1:]
# test_label = data.values[train_num:test_num ,0]

test_data = data.values[0:train_num,1:]
test_label = data.values[0:train_num,0]

print("\n")

print("The kind is surprise_and_sadness:")
print("When the kernel is rbf,the C is 1000")
svc = svm.SVC(kernel = 'rbf', C = 1000) #svm是二分类，svc支持多分类
svc.fit(train_data,train_label)
pre = svc.predict(test_data)
    # 计算准确率
score = svc.score(test_data, test_label)
print(u'准确率：%f' % score)
print("--------------------------")
print("\n")

print("When the kernel is rbf,the C is 100")
svc = svm.SVC(kernel = 'rbf', C = 100) #svm是二分类，svc支持多分类
svc.fit(train_data,train_label)
pre = svc.predict(test_data)
    # 计算准确率
score = svc.score(test_data, test_label)
print(u'准确率：%f' % score)
print("--------------------------")
print("\n")

print("When the kernel is rbf,the C is 10")
svc = svm.SVC(kernel = 'rbf', C = 10) #svm是二分类，svc支持多分类
svc.fit(train_data,train_label)
pre = svc.predict(test_data)
    # 计算准确率
score = svc.score(test_data, test_label)
print(u'准确率：%f' % score)
print("--------------------------")
print("\n")

print("When the kernel is rbf,the C is 1")
svc = svm.SVC(kernel = 'rbf', C = 1) #svm是二分类，svc支持多分类
svc.fit(train_data,train_label)
pre = svc.predict(test_data)
    # 计算准确率
score = svc.score(test_data, test_label)
print(u'准确率：%f' % score)
print("--------------------------")
print("\n")






print("When the kernel is poly,the C is 1000")
svc = svm.SVC(kernel = 'poly', C = 1000) #svm是二分类，svc支持多分类
svc.fit(train_data,train_label)
pre = svc.predict(test_data)
    # 计算准确率
score = svc.score(test_data, test_label)
print(u'准确率：%f' % score)
print("--------------------------")
print("\n")

print("When the kernel is poly,the C is 100")
svc = svm.SVC(kernel = 'poly', C = 100) #svm是二分类，svc支持多分类
svc.fit(train_data,train_label)
pre = svc.predict(test_data)
    # 计算准确率
score = svc.score(test_data, test_label)
print(u'准确率：%f' % score)
print("--------------------------")
print("\n")

print("When the kernel is poly,the C is 10")
svc = svm.SVC(kernel = 'poly', C = 10) #svm是二分类，svc支持多分类
svc.fit(train_data,train_label)
pre = svc.predict(test_data)
    # 计算准确率
score = svc.score(test_data, test_label)
print(u'准确率：%f' % score)
print("--------------------------")
print("\n")

print("When the kernel is poly,the C is 1")
svc = svm.SVC(kernel = 'poly', C = 1) #svm是二分类，svc支持多分类
svc.fit(train_data,train_label)
pre = svc.predict(test_data)
    # 计算准确率
score = svc.score(test_data, test_label)
print(u'准确率：%f' % score)
print("--------------------------")








