import numpy as np
import libmr
import sklearn.metrics.pairwise
import time
from contextlib import contextmanager
from multiprocessing import Pool,cpu_count
import itertools as it
import pandas as pd
import config


@contextmanager
def timer(message):
    """
    测试运行时间
    """
    print(message)
    start = time.time()
    yield
    stop = time.time()
    print("...elapsed time: {}".format(stop-start))


def euclidean_cdist(X,Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="euclidean", n_jobs=1)
def euclidean_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="euclidean", n_jobs=1)
def cosine_cdist(X,Y):
    return sklearn.metrics.pairwise.pairwise_distances(X, Y, metric="cosine", n_jobs=1)
def cosine_pdist(X):
    return sklearn.metrics.pairwise.pairwise_distances(X, metric="cosine", n_jobs=1)

dist_func_lookup = {
    "cosine":{"cdist":cosine_cdist,
              "pdist":cosine_pdist},
    
    "euclidean":{"cdist":euclidean_cdist,
                 "pdist":euclidean_pdist}
}



cdist_func = dist_func_lookup["euclidean"]["cdist"]
pdist_func = dist_func_lookup["euclidean"]["pdist"]


def set_cover_greedy(universe,subsets,cost=lambda x:1.0):
    """
    贪心覆盖算法，论文所涉及到的
    """
    universe = set(universe)

    subsets = list(map(set,subsets))
    #[{0, 258, 3, 388, 518, 522, 523, 398},{},,,{}]

    covered = set()
    cover_indices = []
    while covered != universe:
        max_index = (np.array([len(x - covered) for x in subsets])).argmax()

        covered |= subsets[max_index]  #{512, 2, 259, 271, 19, 533, 537, 282, 283, 541, 548, 552, 554, 300, 47, 561}

        cover_indices.append(max_index) #max_index是EVs
    return cover_indices

def set_cover(points,weibulls,solver=set_cover_greedy):
    """
    集合封面的通用包装。采用解算器函数。可以进行线性规划近似，但默认的贪婪方法在多项式时间内有界
    """
    universe = list(range(len(points)))  #这里用的是universe [0, 1....Nl]代表的是0到该类总样本数的范围

    d_mat = pdist_func(points)   #计算距离mij 这里的距离是ij属于同一类别

    p = Pool(cpu_count())
    probs = np.array(p.map(weibull_eval_parallel,list(zip(d_mat,weibulls))))   #xi到其他点的距离（同一类别）在xi韦布分布上的概率   将距离转化为概率

    p.close()
    p.join()
    thresholded = list(zip(*np.where(probs >= cover_threshold)))
    #冗余阈值，挑选出符合概率大于冗余阈值的所有点，也就是挑选出所有的EVs
    #[  0   0   0 ... 571 571 571] [  0   3  20 ... 552 554 571]
    #[(0, 0), (0, 3), (0, 20), (0, 24), (0, 76),

    subsets = {k:tuple(set(x[1] for x in v)) for k,v in it.groupby(thresholded, key=lambda x:x[0])}
    #{0: (0, 462), 1: (1,), 2: (2,), 3: (3,), 4: (4,), 5: (513, 446, 5, 102, 461, 110, 310, 62), }

    subsets = [subsets[i] for i in universe]
    #(0, 258, 3, 388, 518, 522, 523),(),()
    keep_indices = solver(universe,subsets)
    #[363, 118, 12, 158, 237, 59, 327, 242, 164, 102, 215, 88, 440, 83]

    return keep_indices

def reduce_model(points,weibulls,labels,labels_to_reduce=None):
    """
    模型减少操作
    """
    global cover_threshold  #冗余阈值
    cover_threshold = config.cover_threshold
    if cover_threshold >= 1.0:
        # optimize for the trivial case
        return points,weibulls,labels  #不需要去减少模型
    ulabels = np.unique(labels)  #[ 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20]   pos的标签

    if labels_to_reduce == None:
        labels_to_reduce = ulabels
    labels_to_reduce = set(labels_to_reduce) #{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}

    keep = np.array([],dtype=int) #[]

    for ulabel in ulabels:
        ind = np.where(labels == ulabel) #(array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,] y=1时 的行数

        if ulabel in labels_to_reduce: 
            print(("...reducing model for label {}".format(ulabel)))
            keep_ind = set_cover(points[ind],[weibulls[i] for i in ind[0]])

            keep = np.concatenate((keep,ind[0][keep_ind]))  #找出EVs

        else:
            keep = np.concatenate((keep,ind[0]))
    points = points[keep]

    weibulls = [weibulls[i] for i in keep]
    labels = labels[keep]
    return points,weibulls,labels

def weibull_fit_parallel(args):
    """韦布尔拟合概率"""
    global tailsize
    tailsize = config.tailsize
    dists,row,labels = args

    nearest = np.partition(dists[np.where(labels != labels[row])],tailsize)
    """
    dists  为[4.         5.5        5.33853913 ... 6.2249498  4.0620192  7.07106781]  是xi对应于其他点的距离大小
    labels 为[ 1  1  1 ... 20 20 20]  为标签
    row  为i下标
    nearest  结果就是说是取xi的最小半距离的前tailsize个最近的点（不同类别）
    """

    mr = libmr.MR()
    mr.fit_low(nearest,tailsize)



    return str(mr)

def weibull_eval_parallel(args):

    dists,weibull_params = args  #xi到其他点的距离（同一类别），xi韦布分布

    mr = libmr.load_from_string(weibull_params)
    probs = mr.w_score_vector(dists) #xi到其他点的距离（同一类别）在xi韦布分布上的概率   将距离转化为概率

    return probs

def fuse_prob_for_label(prob_mat,num_to_fuse):

    return np.average(np.partition(prob_mat,-num_to_fuse,axis=0)[-num_to_fuse:,:],axis=0)

def fit(X,y):

    global margin_scale  #距离乘以一个比例，这个比例大小在实验中是0.5，也就是最小半距离
    margin_scale = config.margin_scale
    d_mat = margin_scale*pdist_func(X)   #注意，这里用的是欧式距离，欧式距离的话也就是向量的内积，求出最小半距离

    p = Pool(cpu_count())
    row_range = list(range(len(d_mat))) #(0-10773)
    args = list(zip(d_mat,row_range,[y for i in row_range]))  #(第一行距离，第一行为0，y标签)

    with timer("...getting weibulls"):
        weibulls = p.map(weibull_fit_parallel, args)
    p.close()
    p.join()
    return weibulls

def predict(X,points,weibulls,labels):
    """
    预测.未知类为99
    """
    global num_to_fuse,ot
    num_to_fuse= config.num_to_fuse
    ot= config.ot
    d_mat = cdist_func(points,X).astype(np.float64)  #计算x，y的距离
    p = Pool(cpu_count())
    probs = np.array(p.map(weibull_eval_parallel,list(zip(d_mat,weibulls))))  #xy的距离和x模型，计算概率
    p.close()
    p.join()
    ulabels = np.unique(labels)
    fused_probs = []
    for ulabel in ulabels:

        fused_probs.append(fuse_prob_for_label(probs[np.where(labels == ulabel)],num_to_fuse))
    fused_probs = np.array(fused_probs)
    max_ind = np.argmax(fused_probs,axis=0)
    # print("********")
    # print(max_ind)
    predicted_labels = ulabels[max_ind]
    # print("********")
    # print(predicted_labels)
    confidence = fused_probs[max_ind]
    # print("********")
    # print(confidence)
    # ** ** ** **
    # [0 0 0... 0 9 0]
    # ** ** ** **
    # [1  1  1...  1 10  1]
    for i in range(confidence.shape[0]):
          if(confidence[i,i] < ot):
            predicted_labels[i] = 99  #if probability threshold is less than the specified value then it ia labelled as 99 value
    return predicted_labels,fused_probs


def load_data(fname):
    df = pd.read_csv(fname,header = None)
    labels = df.iloc[:,0]
    data = df.iloc[:,1:]
    return np.array(data),np.array(labels)

def get_accuracy(predictions,labels):
    return sum(predictions == labels)/float(len(predictions))

def update_params(n_tailsize,
                  n_cover_threshold,
                  n_cdist_func,
                  n_pdist_func,
                  n_num_to_fuse,
                  n_margin_scale):
    global tailsize,cover_threshold,cdist_func,pdist_func,num_to_fuse,margin_scale
    tailsize = n_tailsize
    cover_threshold = n_cover_threshold
    cdist_func = n_cdist_func
    pdist_func = n_pdist_func
    num_to_fuse = n_num_to_fuse
    margin_scale= n_margin_scale


def open_set_evm(train_fname,test_fname):

    with timer("...loading train data"):
       Xtrain,ytrain = load_data(train_fname)  
       print(Xtrain.shape,ytrain.shape)         # √  加载训练数据，这里用的是前15个字母当为已知类，数据规模是1万
    with timer("...loading test data"):
        Xtest, ytest = load_data(test_fname)
        print(Xtest.shape,ytest.shape)          # √加载测试数据，数据规模是6000
    with timer("...fitting train set"):
        weibulls = []
        weibulls = fit(Xtrain,ytrain)    #用极小值分布去拟合训练数据，韦布尔分布
    with timer("...reducing model"):
        Xtrain,weibulls,ytrain = reduce_model(Xtrain,weibulls,ytrain)  #模型减少操作
    print(("...model size: {}".format(len(ytrain))))
    with timer("...getting predictions"):
        predictions,probs = predict(Xtest,Xtrain,weibulls,ytrain)
    with timer("...evaluating predictions"):
        accuracy = get_accuracy(predictions,ytest)       
    print("accuracy: {}".format(accuracy))
    return accuracy,predictions,ytest

if __name__ == '__main__':
    accuracy, predictions, yactual = open_set_evm('../letter/train.csv','../letter/test.csv')


