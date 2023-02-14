#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6


import numpy as np
from torchvision import datasets, transforms
import pdb
import torch
from options import args_parser

args = args_parser()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)


def mnist_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def mnist_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # 60,000 training imgs -->  200 imgs/shard X 300 shards
    num_shards, num_imgs = 200, 300
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign 2 shards/client
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def mnist_noniid_unequal(dataset, num_users):
    """
    Sample non-I.I.D client data from MNIST dataset s.t clients
    have unequal amount of data
    :param dataset:
    :param num_users:
    :returns a dict of clients with each clients assigned certain
    number of training imgs
    """
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    num_shards, num_imgs = 1200, 50
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    labels = dataset.train_labels.numpy()

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30

    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard+1,
                                          size=num_users)
    random_shard_size = np.around(random_shard_size /
                                  sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)

    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:

        for i in range(num_users):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        random_shard_size = random_shard_size-1

        # Next, randomly assign the remaining shards
        for i in range(num_users):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)
    else:

        for i in range(num_users):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_users, key=lambda x: len(dict_users.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size,
                                            replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_users[k] = np.concatenate(
                    (dict_users[k], idxs[rand*num_imgs:(rand+1)*num_imgs]),
                    axis=0)

    return dict_users


def cifar_iid(dataset, num_users):
    """
    Sample I.I.D. client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def cifar_noniid(dataset, num_users):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """
    num_shards, num_imgs = 200, 250
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}
    idxs = np.arange(num_shards*num_imgs)
    # labels = dataset.train_labels.numpy()
    #labels = np.array(dataset.train_labels)
    labels = np.array(dataset.targets)

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_users[i] = np.concatenate(
                (dict_users[i], idxs[rand*num_imgs:(rand+1)*num_imgs]), axis=0)
    return dict_users


def cifar_noniid_new(dataset, num_users, Lam_high, Num_chunk):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_users:
    :return:
    """

    num_class = 10

    def cal_pmatrix(Num_leave, Num_locate, Lam=Lam_high):
        P_matrix = np.random.rand(int(Num_leave))
        P_matrix = P_matrix/np.sum(P_matrix)
        # print(P_matrix)
        P_add = np.random.rand(int(Num_leave/10))
        P_add = P_add/sum(P_add)*(0.9*(Lam/(1-Lam))-0.1)
        P_matrix[int(Num_locate*(Num_leave/10)):int(Num_locate*(Num_leave/10))+int(Num_leave/10)
                 ] = P_matrix[int(Num_locate*(Num_leave/10)):int(Num_locate*(Num_leave/10))+int(Num_leave/10)]+P_add
        P_matrix = P_matrix/(0.9*(1/(1-Lam)))
        return P_matrix

    num_chunk = Num_chunk
    num_shards, num_imgs = int(
        num_users*num_chunk), int(50000/(num_users*num_chunk))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}

    labels = np.array(dataset.targets)
    print(len(labels))
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        num_locate = i % num_class
        num_leave = len(idx_shard)
        p_matrix = cal_pmatrix(Num_leave=num_leave, Num_locate=num_locate)
        # print(sum(p_matrix))
        rand_set = set(np.random.choice(
            idx_shard, num_chunk, p=p_matrix, replace=False))
        # pdb.set_trace()

        idx_shard = list(set(idx_shard) - rand_set)
        jf = 0
        for rand in rand_set:
            if jf < int(num_chunk*(500/num_chunk-int(500/num_chunk))):
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs+1]), axis=0)
            else:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            jf = jf+1
    return dict_users


def kvasir_noniid_new(dataset, num_users, Lam_high, Num_chunk):
    """
    Sample non-I.I.D client data from kvasir dataset
    :param dataset:
    :param num_users:
    :return:
    """

    num_class = 8

    def cal_pmatrix_kvasir(Num_leave, Num_locate, Lam=Lam_high):
        P_matrix = np.random.rand(int(Num_leave))
        P_matrix = P_matrix/np.sum(P_matrix)
        P_add = np.random.rand(int(Num_leave/8))
        P_add = P_add/sum(P_add)*(0.875*(Lam/(1-Lam))-0.125)
        P_matrix[int(Num_locate*(Num_leave/8)):int(Num_locate*(Num_leave/8))+int(Num_leave/8)
                 ] = P_matrix[int(Num_locate*(Num_leave/8)):int(Num_locate*(Num_leave/8))+int(Num_leave/8)]+P_add
        P_matrix = P_matrix/(0.875*(1/(1-Lam)))
        return P_matrix

    num_chunk = Num_chunk
    num_shards, num_imgs = int(
        num_users*num_chunk), int(6000/(num_users*num_chunk))
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}

    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    for i in range(num_users):
        num_locate = i % num_class
        num_leave = len(idx_shard)
        p_matrix = cal_pmatrix_kvasir(
            Num_leave=num_leave, Num_locate=num_locate)

        rand_set = set(np.random.choice(
            idx_shard, num_chunk, p=p_matrix, replace=False))

        idx_shard = list(set(idx_shard) - rand_set)
        jf = 0
        for rand in rand_set:
            if jf < int(num_chunk*(120/num_chunk-int(120/num_chunk))):
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs+1]), axis=0)
            else:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            jf = jf+1
    return dict_users


def covid_iid(dataset, num_users):
    """
    Sample I.I.D. client data from covid dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(len(dataset))]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items,
                                             replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users


def covid_noniid(dataset, num_users, Lam_high, Num_chunk):
    """
    Sample non-I.I.D client data from covid dataset
    :param dataset:
    :param num_users:
    :return:
    """
    # the probability matrix we need
    num_class = 3
    data_class_weight = [7966/13942, 5469/13942, 507/13942]
    accumulated_data_class_weight = [0, 7966/13942, 7966/13942+5469/13942, 1]

    def cal_pmatrix_covid(Num_leave, Num_locate, Lam=Lam_high):
        P_matrix = np.ones(int(Num_leave))
        P_matrix = P_matrix/np.sum(P_matrix)
        P_add = Lam * \
            np.ones(
                int(Num_leave*data_class_weight[Num_locate]))/np.sum(P_matrix)
        P_matrix[int(Num_leave*accumulated_data_class_weight[Num_locate]):int(Num_leave*accumulated_data_class_weight[Num_locate])+int(Num_leave*data_class_weight[Num_locate])
                 ] = P_matrix[int(Num_leave*accumulated_data_class_weight[Num_locate]):int(Num_leave*accumulated_data_class_weight[Num_locate])+int(Num_leave*data_class_weight[Num_locate])]+P_add
        P_matrix = P_matrix/np.sum(P_matrix)
        return P_matrix

    num_chunk = Num_chunk
    num_shards, num_imgs = int(
        num_users*num_chunk), int(13942/(num_users*num_chunk))

    loc_int = -(13942-num_shards*num_imgs)
    idx_shard = [i for i in range(num_shards)]
    dict_users = {i: np.array([]) for i in range(num_users)}

    labels = np.array(dataset.targets)
    idxs = np.arange(len(labels))

    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels[:, :loc_int] = idxs_labels[:,
                                           idxs_labels[1, :loc_int].argsort()]
    idxs = idxs_labels[0, :]

    # divide and assign
    jf = 0
    for i in range(num_users):
        num_locate = i % num_class
        num_leave = len(idx_shard)
        p_matrix = cal_pmatrix_covid(
            Num_leave=num_leave, Num_locate=num_locate)
        rand_set = set(np.random.choice(
            idx_shard, num_chunk, p=p_matrix, replace=False))

        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            if jf < -loc_int:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[-jf-2:-jf-1]), axis=0)
            else:
                dict_users[i] = np.concatenate(
                    (dict_users[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
            jf = jf+1
    return dict_users


if __name__ == '__main__':
    dataset_train = datasets.MNIST('./data/mnist/', train=True, download=True,
                                   transform=transforms.Compose([
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,),
                                                            (0.3081,))
                                   ]))
    num = 100
    d = mnist_noniid(dataset_train, num)
