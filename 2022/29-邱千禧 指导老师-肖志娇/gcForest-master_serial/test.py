import gzip
import os
import struct

from GCForest import GCForest

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import wave
from sklearn import preprocessing
import datetime

from sklearn import datasets

# 扫描方法不对，应该是横着竖着都扫，不是对角线扫描
#参数，参考一下论文的超参数设置

"""# loading the data
iris = load_iris()
X = iris.data
y = iris.target"""




score=[]

def mytry(data_type="serial",X=None,y=None,dataname="220_datasets_4_plus.npz",scan_type="each"):
    if data_type=="serial":
        dataname = dataname
        datasets = np.load(dataname)
        X = datasets['data']
        y = datasets['target']
        # 对y进行标签编码（随机森林会分类成0,1,2...的结果，为了计算精度，需要用标签编码）
        my_label = preprocessing.LabelEncoder()
        y = my_label.fit_transform(y)

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.33)

    gcf = GCForest(windowRatio=[1/16, 1/8, 1/4],data_type=data_type, tolerance=0.0, scan_type=scan_type)
    gcf.fit(X_tr, y_tr)

    """pred_X = gcf.predict(X_tr)
    accuracy = accuracy_score(y_true=y_tr, y_pred=pred_X)
    print('train gcForest accuracy : {}'.format(accuracy))"""

    pred = gcf.predict(X_te)
    print(pred)

    accuracy = accuracy_score(y_true=y_te, y_pred=pred)
    print('test gcForest accuracy : {}'.format(accuracy))









"""digits = datasets.load_digits()
x=digits.images
y=digits.target
print("--------------------------------------------------------digit_dataset--------------------------------------------------------")
print("-------------------------------each in below----------------------------------")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
mytry(data_type="image",X=x,y=y,scan_type="each")
mytry(data_type="image",X=x,y=y,scan_type="each")
mytry(data_type="image",X=x,y=y,scan_type="each")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
print("-------------------------------all in below----------------------------------")
mytry(data_type="image",X=x,y=y,scan_type="all")
mytry(data_type="image",X=x,y=y,scan_type="all")
mytry(data_type="image",X=x,y=y,scan_type="all")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)"""


def load_mnist_train(images_path,labels_path):
    #使用gzip打开文件
    with gzip.open(labels_path, 'rb') as lbpath:
        #使用struct.unpack方法读取前两个数据，>代表高位在前，I代表32位整型。lbpath.read(8)表示一次从文件中读取8个字节
        #这样读到的前两个数据分别是magic number和样本个数
        magic, n = struct.unpack('>II',lbpath.read(8))
        #使用np.fromstring读取剩下的数据，lbpath.read()表示读取所有的数据
        labels = np.fromstring(lbpath.read(),dtype=np.uint8)
    with gzip.open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = np.fromstring(imgpath.read(),dtype=np.uint8).reshape(len(labels), 784)
    return images, labels


x_train,y_train=load_mnist_train("./genres/MNIST/train-images-idx3-ubyte.gz","./genres/MNIST/train-labels-idx1-ubyte.gz")
x_test,y_test=load_mnist_train("./genres/MNIST/t10k-images-idx3-ubyte.gz","./genres/MNIST/t10k-labels-idx1-ubyte.gz")
x_train=x_train.reshape(60000,28,-1)
x_test=x_test.reshape(10000,28,-1)
X = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

print("--------------------------------------------------------MNIST_dataset--------------------------------------------------------")
print("-------------------------------each in below----------------------------------")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
mytry(data_type="image",X=X,y=y,scan_type="each")
mytry(data_type="image",X=X,y=y,scan_type="each")
mytry(data_type="image",X=X,y=y,scan_type="each")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
print("-------------------------------all in below----------------------------------")
mytry(data_type="image",X=X,y=y,scan_type="all")
mytry(data_type="image",X=X,y=y,scan_type="all")
mytry(data_type="image",X=X,y=y,scan_type="all")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)



"""print("--------------------------------------------------------220_datasets--------------------------------------------------------")
print("-------------------------------each in below----------------------------------")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
mytry(data_type="serial",dataname="220_datasets.npz",scan_type="each")
mytry(data_type="serial",dataname="220_datasets.npz",scan_type="each")
mytry(data_type="serial",dataname="220_datasets.npz",scan_type="each")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
print("-------------------------------all in below----------------------------------")
mytry(data_type="serial",dataname="220_datasets.npz",scan_type="all")
mytry(data_type="serial",dataname="220_datasets.npz",scan_type="all")
mytry(data_type="serial",dataname="220_datasets.npz",scan_type="all")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)

print("--------------------------------------------------------220_datasets_4_plus--------------------------------------------------------")
print("-------------------------------each in below----------------------------------")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
mytry(data_type="serial",dataname="220_datasets_4_plus.npz",scan_type="each")
mytry(data_type="serial",dataname="220_datasets_4_plus.npz",scan_type="each")
mytry(data_type="serial",dataname="220_datasets_4_plus.npz",scan_type="each")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
print("-------------------------------all in below----------------------------------")
mytry(data_type="serial",dataname="220_datasets_4_plus.npz",scan_type="all")
mytry(data_type="serial",dataname="220_datasets_4_plus.npz",scan_type="all")
mytry(data_type="serial",dataname="220_datasets_4_plus.npz",scan_type="all")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)



print("--------------------------------------------------------220_datasets_7_plus_old--------------------------------------------------------")
print("-------------------------------all in below----------------------------------")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
mytry(data_type="serial",dataname="220_datasets_7_plus_old.npz",scan_type="all")
mytry(data_type="serial",dataname="220_datasets_7_plus_old.npz",scan_type="all")
mytry(data_type="serial",dataname="220_datasets_7_plus_old.npz",scan_type="all")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)
print("-------------------------------each in below----------------------------------")
mytry(data_type="serial",dataname="220_datasets_7_plus_old.npz",scan_type="each")
mytry(data_type="serial",dataname="220_datasets_7_plus_old.npz",scan_type="each")
mytry(data_type="serial",dataname="220_datasets_7_plus_old.npz",scan_type="each")
time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(time)"""












