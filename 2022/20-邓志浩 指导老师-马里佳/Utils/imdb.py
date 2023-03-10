# -*- encoding: utf-8 -*-
import dgl
import numpy as np
import pickle

from Utils.multihot2index import multihot2index

"""
This dataset is released by "Graph Transformer Networks" 
 4661 movies (M), 5841 actors (A), and 2270 directors (D).
"""

def get_binary_mask(total_size, indices):
    mask = np.zeros(total_size)
    mask[indices] = 1
    return mask.astype(np.bool)

def load_imdb(onehot_for_nofeature=True, node_types="mda"):
    data_path = "D:/学习/论文/超图/CIAH-main/Dataset\IMDB/"
    with open(data_path + 'node_features.pkl', 'rb') as f:
        node_features = pickle.load(f)
    with open(data_path + 'edges.pkl', 'rb') as f:
        edges = pickle.load(f)
    with open(data_path + 'labels.pkl', 'rb') as f:
        labels = pickle.load(f)
    splits = (4661, 6931, 12772)  # M 0~4661, D: 4661~6931, A: 6931~12772

    edges = [e.nonzero() for e in edges]
    hg = dgl.heterograph({
        ('movie', 'me', 'hyperedge'): np.eye(splits[0], dtype=int).nonzero(),
        ('hyperedge', 'em', 'movie'): np.eye(splits[0], dtype=int).transpose().nonzero(),
        # P和超边等价
        ('actor', 'ae', 'hyperedge'): (edges[3][0]-splits[1], edges[3][1]),
        ('hyperedge', 'ea', 'actor'): (edges[2][0], edges[2][1]-splits[1]),
        ('director', 'de', 'hyperedge'): (edges[1][0]-splits[0], edges[1][1]),
        ('hyperedge', 'ed', 'director'): (edges[0][0], edges[0][1]-splits[0]),
    })
    print(hg)

    m_features = node_features[0: splits[0]]
    d_features = node_features[splits[0]: splits[1]]
    a_features = node_features[splits[1]: splits[2]]
    m_features = multihot2index(m_features)  # 把特征向量变小
    d_features = multihot2index(d_features)
    a_features = multihot2index(a_features)
    bias = max([m_features.max(), d_features.max(), a_features.max()]) + 1
    if onehot_for_nofeature:
        e_features = np.arange(bias, hg.nodes('hyperedge').shape[0] + bias)[:, np.newaxis]
        bias += hg.nodes('hyperedge').shape[0]
    else:
        # 补一个padding
        e_features = np.zeros(hg.nodes('hyperedge').shape[0])[:, np.newaxis]

    features_dict = {
        "e_features": e_features,
        "m_features": m_features,
        "d_features": d_features,
        "a_features": a_features,
    }

    E4N_adjs, features, feature_types = [], [features_dict['e_features']], ["e_features",]
    for node_type in ['m', 'd', 'a']:   # 保持顺序
        if node_type not in node_types:
            continue
        E4N_adjs.append(hg.adj(etype='e{}'.format(node_type), scipy_fmt="coo", transpose=True))  # 邻接矩阵M*M D*M A*M
        feature_types.append("{}_features".format(node_type))
        features.append(features_dict[feature_types[-1]])

    # test = labels[0]
    # test1 = np.array(labels[0])
    train_node = np.array(labels[0])[:, 0]  # 节点
    train_target = np.array(labels[0])[:, 1]  # 类别
    valid_node = np.array(labels[1])[:, 0]
    valid_target = np.array(labels[1])[:, 1]
    test_node = np.array(labels[2])[:, 0]
    test_target = np.array(labels[2])[:, 1]

    labels = np.ones(shape=splits[0], dtype=np.int64) * (-1)
    labels[train_node.astype(np.int64)] = train_target
    labels[valid_node.astype(np.int64)] = valid_target
    labels[test_node.astype(np.int64)] = test_target

    num_classes = train_target.max() + 1

    train_idx = train_node
    val_idx = valid_node
    test_idx = test_node

    num_labeled_nodes = hg.number_of_nodes('hyperedge')
    train_mask = get_binary_mask(num_labeled_nodes, train_idx)
    val_mask = get_binary_mask(num_labeled_nodes, val_idx)
    test_mask = get_binary_mask(num_labeled_nodes, test_idx)

    return E4N_adjs, features, labels, num_classes, feature_types, \
           train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask


if __name__ == '__main__':
    import os

    # os.getcwd()
    os.chdir("../")

    E4N_adjs, features, labels, num_classes, feature_types, \
    train_idx, val_idx, test_idx, \
    train_mask, val_mask, test_mask = load_imdb()