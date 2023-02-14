import numpy as np
from torchvision import datasets, transforms


def mnist_iid(dataset, num_users):


    num_items = int(len(dataset)/num_users)  # 每个用户数据量
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):  # 对于每个用户
        # set() 函数创建一个无序不重复元素集，可进行关系测试，删除重复数据，还可以计算交集、差集、并集等。
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))  # 从all_idxs中采样，生成 单用户数据量的无重复的 一个字典，作为dict_users的第（i+1）个元素
        all_idxs = list(set(all_idxs) - dict_users[i])      # 取差集，删去已经被分配好的数据，直至每个用户都被分配了等量的iid数据
    data = {i:[] for i in range(num_users)}
    for i in range(num_users):
        for j in dict_users[i]:
            data[i].append(dataset[j])
    return data


def mnist_noniid(dataset, num_users):

    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([], dtype='int64') for i in range(num_users)}  # 初始化字典dict_users
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
#     沿着第一个轴堆叠数组。
#     语法格式：numpy.vstack(tup)
#       参数：
#       tup：ndarrays数组序列，如果是一维数组进行堆叠，则数组长度必须相同；除此之外，其它数组堆叠时，除数组第一个轴的长度可以不同，其它轴长度必须一样。

    # y = x.argsort() 将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
    idxs_labels = idxs_labels[:,idxs_labels[1,:].argsort()]
    idxs = idxs_labels[0,:]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            # concatenate进行矩阵拼接
            dict_users[i] = np.concatenate((dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    data = {i:[] for i in range(num_users)}
    for i in range(num_users):
        for j in dict_users[i]:
            data[i].append(dataset[j])
    return data



def cifar_iid(dataset, num_users):
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    data = {i:[] for i in range(num_users)}
    for i in range(num_users):
        for j in dict_users[i]:
            data[i].append(dataset[j])
    return data
