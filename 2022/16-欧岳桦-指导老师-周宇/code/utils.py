import math
import os
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import torch
from scipy import stats
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
from tqdm import trange


class myDataset(InMemoryDataset):
    def __init__(self, root='/data_750+9', dataset='drug_sideEffect_data',
                 drug_simles=None, frequencyMat=None,
                 transform=None, pre_transform=None, simle_graph=None, saliency_map=False):

        super(myDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.saliency_map = saliency_map

        if os.path.isfile(self.processed_paths[0]):
            print('Pre_processed data found: {}, loading...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre_processing...'.format(self.processed_paths[0]))
            self.process(drug_simles, frequencyMat, simle_graph)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):  # 返回一个包含没有处理的数据的名字的list
        pass
        # return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):  # 返回一个包含所有处理过的数据名字的list
        return [self.dataset + '.pt']

    def download(self):  # 下载数据集函数，不需要的话直接填充pass
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drug_silmes, frequencyMat, simle_graph):
        assert (len(drug_silmes) == len(frequencyMat)), "The two lists must be the same L!"
        data_list = []
        data_len = len(drug_silmes)
        print(data_len)
        data_len = trange(data_len)
        data_len.set_description("Processing ")
        for i in data_len:
            smiles = drug_silmes[i]
            labels = frequencyMat[i]
            c_size, features, edge_index, edge_type = simle_graph[smiles]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                edge_index=torch.LongTensor(edge_index),
                                y=torch.FloatTensor([labels]))
            GCNData.__setitem__('edge_type', torch.IntTensor(edge_type * 2 ).flatten())
            # 记录此特征矩阵x的行开始的坐标，为0；
            # 利用DataLoader读取时，返回一个(1 * batch_size)维度的tensor，代表共batch_size个x,每个x的行从x_index[i]开始
            GCNData.__setitem__('x_index', torch.LongTensor([0]))

            # 记录此SMILES对应在所有SMILES的坐标，用于计算loss时查找对应的frequencyMat的位置
            # 利用DataLoader读取时，返回一个(batch_size * 1)的二维列表
            GCNData.index = [i]  # 输出为二维列表

            # 记录每张smile_graph的原子的个数，即特征矩阵x的行数；
            # 利用DataLoader读取时，返回一个(1 * batch_size)维度的tensor，代表共batch_size个x,每个x有c_size[i]的原子
            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))

            data_list.append(GCNData)
            # print(data_list)
        # print(data_list[0])
        # 判断数据对象是否应该保存
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        # 保存到磁盘前进行转化
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        # 将数据对象的python列表整理为内部存储格式，torch_geometric.data_WS.InMemoryDataset
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        pass


def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean())
    return rmse


def mse(y, f):
    mse = ((y - f) ** 2).mean()
    return mse


def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp


def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs


def MAE(y, f):
    rs = sklearn.metrics.mean_absolute_error(y, f)
    return rs


def ci(y, f):
    ind = np.argsort(y)  # argsort函数返回的是数组值从小到大的索引值
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci


def draw_loss(train_losses, test_losses, title, result_folder):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def draw_pearson(pearsons, title, result_folder):
    plt.figure()
    plt.plot(pearsons, label='test pearson')
    plt.ylim((-0.1, 1))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_loss(train_losses, title, result_folder):
    x = [i for i in range(1, len(train_losses) + 1)]
    plt.figure()
    plt.plot(x, train_losses, label='train loss')
    # plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")  # should before show method


def my_draw_pearson(pearsons, title, result_folder):
    plt.figure()
    plt.plot(pearsons, label='test pearson')
    plt.ylim((-0.1, 1))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")


def my_draw_mse(mse, rmse, title, result_folder):
    plt.figure()
    plt.plot(mse, label='test MSE')
    plt.plot(rmse, label='test rMSE')
    plt.ylim((0, 10))
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    # save image
    plt.savefig(result_folder + '/' + title + ".png")


def evaluate_others(M, Tr_neg, Te, positions=[1, 5, 10, 15]):
    """
    :param M: 预测值
    :param Tr_neg: dict， 包含Te
    :param Te:  dict
    :param positions:
    :return:
    """
    prec = np.zeros(len(positions))
    rec = np.zeros(len(positions))
    map_value, auc_value, ndcg = 0.0, 0.0, 0.0
    for u in Te:
        val = M[u, :]
        inx = np.array(Tr_neg[u])
        A = set(Te[u])
        B = set(inx) - A
        # compute precision and recall
        ii = np.argsort(val[inx])[::-1][:max(positions)]
        prec += precision(Te[u], inx[ii], positions)
        rec += recall(Te[u], inx[ii], positions)
        ndcg_user = nDCG(Te[u], inx[ii], 10)
        # compute map and AUC
        pos_inx = np.array(list(A))
        neg_inx = np.array(list(B))
        map_user, auc_user = map_auc(pos_inx, neg_inx, val)
        ndcg += ndcg_user
        map_value += map_user
        auc_value += auc_user
    return map_value / len(Te.keys()), auc_value / len(Te.keys()), ndcg / len(Te.keys()), prec / len(
        Te.keys()), rec / len(Te.keys())


def precision(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set))/float(N)
    elif isinstance(N, list):
        return np.array([precision(actual, predicted, n) for n in N])


def recall(actual, predicted, N):
    if isinstance(N, int):
        inter_set = set(actual) & set(predicted[:N])
        return float(len(inter_set))/float(len(set(actual)))
    elif isinstance(N, list):
        return np.array([recall(actual, predicted, n) for n in N])


def nDCG(Tr, topK, num=None):
    if num is None:
        num = len(topK)
    dcg, vec = 0, []
    for i in range(num):
        if topK[i] in Tr:
            dcg += 1/math.log(i+2, 2)
            vec.append(1)
        else:
            vec.append(0)
    vec.sort(reverse=True)
    idcg = sum([vec[i]/math.log(i+2, 2) for i in range(num)])
    if idcg > 0:
        return dcg/idcg
    else:
        return idcg


def map_auc(pos_inx, neg_inx, val):
    map = 0.0
    pos_val, neg_val = val[pos_inx], val[neg_inx]
    ii = np.argsort(pos_val)[::-1]
    jj = np.argsort(neg_val)[::-1]
    pos_sort, neg_sort = pos_val[ii], neg_val[jj]
    auc_num = 0.0
    for i,pos in enumerate(pos_sort):
        num = 0.0
        for neg in neg_sort:
            if pos<=neg:
                num+=1
            else:
                auc_num+=1
        map += (i+1)/(i+num+1)
    return map/len(pos_inx), auc_num/(len(pos_inx)*len(neg_inx))