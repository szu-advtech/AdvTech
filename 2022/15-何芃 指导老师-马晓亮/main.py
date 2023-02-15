from sklearn import neural_network
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")  # 忽略警告消息


class Naive_Bayes:
    def __init__(self):
        pass

    # 朴素贝叶斯训练过程
    def nb_fit(self, X, y):
        classes = y[y.columns[0]].unique()
        class_count = y[y.columns[0]].value_counts()  # 计数函数
        # 类先验概率
        class_prior = class_count / len(y)

        # 计算类条件概率
        prior = dict()
        for col in X.columns:
            for j in classes:
                p_x_y = X[(y == j).values][col].value_counts()
                for i in p_x_y.index:
                    prior[(col, i, j)] = p_x_y[i] / class_count[j]
        self.classes = classes
        self.class_prior = class_prior
        self.prior = prior


    # 预测新的实例
    def predict(self, X_test, W, x_test_value, m):
        res = []
        for c in self.classes:
            p_y = self.class_prior[c]
            p_x_y = 1
            attindex = 0

            for i in X_test.items():
                tt = x_test_value[attindex] + m * attindex
                if tuple(list(i) + [c]) not in self.prior.keys():
                    p_x_y *= 0
                else:
                    p_x_y *= np.power(self.prior[tuple(list(i) + [c])], W[c, tt])
                attindex += 1
            res.append(p_y * p_x_y)
        return res

    def predict_result(self, X_test, W, k):
        result = []
        for t in range(X_test.shape[0]):
            x_test_value = X_test.iloc[t].values
            test = dict(X_test.iloc[t])
            res = self.predict(test, W, x_test_value, k)
            result.append(res)
        return result

    def pred(self, x_test):
        result = []
        for t in range(x_test.shape[0]):
            res = []
            for c in self.classes:
                p_y = self.class_prior[c]
                p_x_y = 1

                test = dict(x_test.iloc[t])
                for i in test.items():
                    if tuple(list(i) + [c]) not in self.prior.keys():
                        p_x_y *= 0
                    else:
                        p_x_y *= self.prior[tuple(list(i) + [c])]
                res.append(p_y * p_x_y)
            result.append(res)
        return np.array(result)

class MAWNB:
    def __init__(self, x_train, y_train, k, features):
        self.x_train = x_train
        self.y_train = y_train
        self.k = k
        self.features = features

    def raw_view(self, x_train, y_train, k):
        x1 = pd.DataFrame(x_train, columns=self.features)
        y1 = pd.DataFrame(y_train, columns=['class'])
        self.nb_m1 = Naive_Bayes()
        self.nb_m1.nb_fit(x1, y1)
        m = x_train.shape[1]
        x_train = self.nb_m1.pred(x1)
        mlp = neural_network.MLPClassifier(hidden_layer_sizes=(m * k), solver='lbfgs',max_iter=1000)

        mlp.fit(x_train, y_train)

        return mlp.coefs_[0]

    def first_view(self, x_train, y_train, k):
        m = x_train.shape[1]
        x = []
        self.S = []
        for i in range(m):
            clf = BernoulliNB() #由于SPODE跑的太慢，这里换成了BernoulliNB
            #BernoulliNB也是一个基于NBC的分类器，实验效果与原SPODE效果差不多
            clf.fit(x_train,y_train)
            t = clf.predict(x_train)
            t = np.array(t)

            x.append(t)
            self.S.append(clf)
        x_first = x[0].reshape(-1, 1)

        for i in range(1, m):
            x_first = np.concatenate([x_first, x[i].reshape(-1, 1)], 1)


        x1 = pd.DataFrame(x_first, columns=self.features)
        y1 = pd.DataFrame(y_train, columns=['class'])
        self.nb_m2 = Naive_Bayes()
        self.nb_m2.nb_fit(x1, y1)
        x_first = self.nb_m2.pred(x1)

        mlp = neural_network.MLPClassifier(hidden_layer_sizes=(m * k), solver='lbfgs',max_iter=1000)
        mlp.fit(x_first, y_train)
        return mlp.coefs_[0]

    def second_view(self, x_train, y_train, k):
        x = []
        m = x_train.shape[1]
        self.R = []
        for i in range(m):
            clf = RandomForestClassifier()#max_features='log2'
            clf.fit(x_train, y_train)
            rt_pred = clf.predict(x_train)
            self.R.append(clf)
            x.append(rt_pred)
        x_second = x[0].reshape(-1, 1)
        for i in range(1, m):
            x_second = np.concatenate([x_second, x[i].reshape(-1, 1)], 1)

        self.nb_m3 = Naive_Bayes()
        x2 = pd.DataFrame(x_second, columns=self.features)
        y2 = pd.DataFrame(y_train, columns=['class'])
        self.nb_m3.nb_fit(x2, y2)

        x_second = self.nb_m3.pred(x2)
        mlp = neural_network.MLPClassifier(hidden_layer_sizes=(m * k), solver='lbfgs',max_iter=1000)
        mlp.fit(x_second, y_train)

        return mlp.coefs_[0]

    def fit(self):
        self.M1 = self.raw_view(self.x_train, self.y_train, self.k)
        print("M1 !!!")
        self.M2 = self.first_view(self.x_train, self.y_train, self.k)
        print("M2 !!!")
        self.M3 = self.second_view(self.x_train, self.y_train, self.k)
        print("M3 !!!")
    def pred(self, x_test):
        x2 = []
        x3 = []
        for i in range(x_test.shape[1]):
            x2.append(self.S[i].predict(x_test))
            x3.append(self.R[i].predict(x_test))
        x2 = np.asarray(x2)
        x_test_first = x2[0].reshape(-1, 1)
        x_test_second = x3[0].reshape(-1, 1)
        for i in range(1, x_test.shape[1]):
            x_test_first = np.concatenate([x_test_first, x2[0].reshape(-1, 1)], 1)
            x_test_second = np.concatenate([x_test_second, x3[0].reshape(-1, 1)], 1)
        x_test_1 = pd.DataFrame(x_test, columns=self.features)
        x_test_2 = pd.DataFrame(x_test_first, columns=self.features)
        x_test_3 = pd.DataFrame(x_test_second, columns=self.features)
        p1 = np.array(self.nb_m1.predict_result(x_test_1, self.M1, self.k))
        p2 = np.array(self.nb_m2.predict_result(x_test_2, self.M2, self.k))
        p3 = np.array(self.nb_m3.predict_result(x_test_3, self.M3, self.k))

        p = p1 + p2 + p3

        result = [item.tolist().index(max(item.tolist())) for item in p]

        return result, p1, p2, p3

k = 10 #离散为0-9之间的数
data = pd.read_csv('iris1.csv')
print('iris')
# data = pd.read_csv('kr-vs-kp1.csv')
# print('kr-vs-kp')
# data = pd.read_csv('agaricus-lepiota1.csv')
# print('agaricus-lepiota')
Encoder = LabelEncoder()  # 离散型的数据转换成0~n-1之间的整数
t = data.columns  # .columns返回列索引 .index返回索引
features = t[:-1]  # 去掉最后一个字符的结果
for i in range(t.shape[0]):  # shape[0]读取行数，shape[1]读取列数
    data[t[i]] = Encoder.fit_transform(data[t[i]])  # 对数据进行标准化
data = data.values  # 返回一个字典中的所有值
x = data[:, :-1]
y = data[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=72431) #random_state=72431
print(x_train.shape)
# random_state：是随机数的种子。

mawnb = MAWNB(x_train,y_train,k,features)   # 模型创建
mawnb.fit()                                 #模型训练

pred,p1,p2,p3 = mawnb.pred(x_test)
print('测试结果：{:.2f}%'.format(metrics.accuracy_score(y_test,pred) * 100 ))

