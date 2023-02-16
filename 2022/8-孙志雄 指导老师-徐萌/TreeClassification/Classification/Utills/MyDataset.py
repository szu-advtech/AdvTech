"""
@Name: MyDataset.py
@Auth: SniperIN_IKBear
@Date: 2022/12/1-16:42
@Desc: 
@Ver : 0.0.0
"""
import numpy as np
import torch
import hdf5storage as hdf5
import scipy.io as sio


from HybridSN.Utills.Get3DPatch import applyPCA, createImageCubes, splitTrainTestSet

""" Testing dataset"""
class TestDS(torch.utils.data.Dataset):
    def __init__(self,Xtest,ytest):
        self.len = Xtest.shape[0]
        self.x_data = torch.FloatTensor(Xtest)
        self.y_data = torch.LongTensor(ytest)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len
""" Training dataset"""
class TrainDS(torch.utils.data.Dataset):
    def __init__(self,Xtrain,ytrain):
        self.len = Xtrain.shape[0]
        self.x_data = torch.FloatTensor(Xtrain)
        self.y_data = torch.LongTensor(ytrain)
    def __getitem__(self, index):
        # 根据索引返回数据和对应的标签
        return self.x_data[index], self.y_data[index]
    def __len__(self):
        # 返回文件数据的数目
        return self.len

def load_set():
    # X = sio.loadmat('/home2/szx/Dataset/TreeDetection/HybridSN/Indian_pines_corrected.mat')['indian_pines_corrected']
    # y = sio.loadmat('/home2/szx/Dataset/TreeDetection/HybridSN/Indian_pines_gt.mat')['indian_pines_gt']
    X = sio.loadmat('/home2/szx/Dataset/TreeDetection/HybridSN/hyper_lidar_szu.mat')['data']
    y = sio.loadmat('/home2/szx/Dataset/TreeDetection/HybridSN/true_label_szu.mat')['AllTrueLabel']
    # X = X[:,:240,:]
    # y = y[:,:240]
    # X = hdf5.loadmat('/home2/szx/Dataset/TreeDetection/HybridSN/5-2134567-data.mat')['data']
    # y = hdf5.loadmat('/home2/szx/Dataset/TreeDetection/HybridSN/5-2134567-data.mat')['map']
    # 用于测试样本的比例
    # 数据归一化
    def Normalize(data):
        h, w, c = data.shape
        data = data.reshape((h * w, c))
        data -= np.min(data, axis=0)
        data /= np.max(data, axis=0)
        data = data.reshape((h, w, c))
        return data

    # 数据归一化并缩放到[-1, 1]
    X = 2 * Normalize(X) - 1
    test_ratio = 0.50
    # 每个像素周围提取 patch 的尺寸
    patch_size = 25
    # 使用 PCA 降维，得到主成分的数量
    pca_components = 30
    print('Hyperspectral data shape: ', X.shape)
    print('Label shape: ', y.shape)

    print('\n... ... PCA tranformation ... ...')
    X_pca = applyPCA(X, numComponents=pca_components)
    print('Data shape after PCA: ', X_pca.shape)

    print('\n... ... create data cubes ... ...')
    X_pca, y = createImageCubes(X_pca, y, windowSize=patch_size)
    print('Data cube X shape: ', X_pca.shape)
    print('Data cube y shape: ', y.shape)

    print('\n... ... create train & test data ... ...')
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(X_pca, y, test_ratio)
    print('Xtrain shape: ', Xtrain.shape)
    print('Xtest  shape: ', Xtest.shape)

    # 改变 Xtrain, Ytrain 的形状，以符合 keras 的要求
    Xtrain = Xtrain.reshape(-1, patch_size, patch_size, pca_components, 1)
    Xtest = Xtest.reshape(-1, patch_size, patch_size, pca_components, 1)
    print('before transpose: Xtrain shape: ', Xtrain.shape)
    print('before transpose: Xtest  shape: ', Xtest.shape)

    # 为了适应 pytorch 结构，数据要做 transpose
    Xtrain = Xtrain.transpose(0, 4, 3, 1, 2)
    Xtest = Xtest.transpose(0, 4, 3, 1, 2)
    print('after transpose: Xtrain shape: ', Xtrain.shape)
    print('after transpose: Xtest  shape: ', Xtest.shape)

    return Xtrain, Xtest, ytrain, ytest
def predicted_set():
    X = sio.loadmat('/home2/szx/Dataset/TreeDetection/HybridSN/Indian_pines_corrected.mat')['indian_pines_corrected']
    patch_size = 25
    # # 使用 PCA 降维，得到主成分的数量
    pca_components = 30
    X_pca = applyPCA(X, numComponents=pca_components)
    Xpre = torch.from_numpy(X_pca.reshape(-1, 1, 200, 25, 25).astype('Float32'))
    return Xpre