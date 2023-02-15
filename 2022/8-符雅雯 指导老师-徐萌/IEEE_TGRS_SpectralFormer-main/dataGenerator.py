import scipy.io as scio
import numpy as np
import os
from sklearn.decomposition import PCA
import random
import h5py

def data_generator(sampleperclass=5, rate=0, seed=121):
    # 拼接路径
    LABEL_FILE_PATH = '/home2/mzc/SpectralFormer/data/GF5_gt.mat'

    # 载入数据
    label = h5py.File(LABEL_FILE_PATH, 'r')['gf5_gt']
    label = np.transpose(label, (1, 0))

    # index 记录重组
    class_dict = {c :[] for c in np.unique(label).tolist()}  # 类别计数器

    print("GF5_class_list:", class_dict)

    # 遍历 label 数组 取出各像素值的标签
    for h in range(label.shape[0]):
        for w in range(label.shape[1]):
            class_dict[label[h, w]].append((label[h, w], h, w))

    # 打乱
    random.seed(seed)
    for k, v in class_dict.items():
        random.shuffle(v)

    # 计数
    counter = {key: len(value) for key, value in class_dict.items()}  # 分类


    print("counter:", counter)


    # 分为test + train
    train_label, test_label = [] ,[]

    if sampleperclass:
        for key, value in class_dict.items():
            # print("partition:", key)
            train_label.extend(value[:sampleperclass])
            test_label.extend(value[sampleperclass:])
    elif rate:
        for key, value in class_dict.items():
            # print("partition:", key)
            RANGE = round(len(value) * rate)
            train_label.extend(value[:RANGE])
            test_label.extend(value[RANGE:])
    else:
        raise Exception("division error")

    train = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    test = np.zeros((label.shape[0], label.shape[1]), dtype=np.uint8)
    for value in train_label:
        train[value[1], value[2]] = value[0]

    for value in test_label:
        test[value[1], value[2]] = value[0]


    scio.savemat('/home2/mzc/SpectralFormer/data/label.mat', {'TR': train, 'TE': test})

if __name__ == '__main__':
    data_generator(10, 5)