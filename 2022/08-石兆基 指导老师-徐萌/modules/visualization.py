"""
可视化原始数据和合成数据
"""

import os
import numpy as np

from matplotlib import pyplot as plt

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


# plot utils
def plot_scatter(*args, **kwargs):
    # plt.plot(*args, **kwargs)
    # scatter参数分别为x，y的个数
    plt.scatter(*args, **kwargs)


def pca_and_tsne(args):
    # 获取数据 比如stock数据总共有3661组数据，每组数据24个序列，每个序列6个特征
    ori_ts = np.load(args.ori_data_dir)   # (len_data, seq_len, z_dim)
    art_ts = np.load(args.art_data_dir)   # (len_data, seq_len, z_dim)
    len_data = min(len(ori_ts), 1000)   # no more than 1000 points
    # 只取前min(len(ori_te), 1000)条数据
    ori_ts = ori_ts[:len_data, :, :]
    art_ts = art_ts[:len_data, :, :]
    subplots = [231, 232, 233, 234, 235, 236]

    # Plot PCA
    plt.figure(figsize=(16, 12))
    for k in range(min(args.z_dim, 6)):
        # 取第k个特征的所有数据
        ts_ori_k = ori_ts[:, :, k]  # (len_data, seq_len)
        ts_art_k = art_ts[:, :, k]  # (len_data, seq_len)

        # n_components表示降维后的维数（特征数）
        pca = PCA(n_components=2)
        # 用原始数据训练PCA模型
        pca.fit(ts_ori_k)
        # 用训练好的模型返回对原始数据和生成数据降维后的数据
        pca_ori = pca.transform(ts_ori_k)
        pca_art = pca.transform(ts_art_k)

        plt.subplot(subplots[k])
        plt.grid()
        plot_scatter(pca_ori[:, 0], pca_ori[:, 1], color='b', alpha=0.1, label='Original')
        plot_scatter(pca_art[:, 0], pca_art[:, 1], color='r', alpha=0.1, label='Synthetic')
        plt.legend()
        plt.title(f'PCA plots for {args.columns[k]}')
        plt.xlabel('x-pca')
        plt.ylabel('y-pca')
        file_name = f'{args.experiment_name}_pca.png'
        file_dir = os.path.join(args.pics_dir, file_name)
        plt.savefig(file_dir)

    # Plot t-SNE
    plt.figure(figsize=(16, 12))
    for k in range(min(args.z_dim, 6)):
        ts_ori_k = ori_ts[:, :, k]  # (len_data, seq_len)
        ts_art_k = art_ts[:, :, k]  # (len_data, seq_len)
        ts_final_k = np.concatenate((ts_ori_k, ts_art_k), axis=0)

        tsne = TSNE(n_components=2)
        tsne_results = tsne.fit_transform(ts_final_k)

        plt.subplot(subplots[k])
        plt.grid()
        plot_scatter(tsne_results[:len_data, 0],
                     tsne_results[:len_data, 1],
                     color='b', alpha=0.1, label='Original')
        plot_scatter(tsne_results[len_data:, 0],
                     tsne_results[len_data:, 1],
                     color='r', alpha=0.1, label='Synthetic')
        plt.legend()
        plt.title(f't-SNE plots for {args.columns[k]}')
        plt.xlabel('x-tsne')
        plt.ylabel('y-tsne')
        file_name = f'{args.experiment_name}_tsne.png'
        file_dir = os.path.join(args.pics_dir, file_name)
        plt.savefig(file_dir)

    # Plot PCA and t-SNE in TimeGAN style
    ori_ts = np.mean(ori_ts, axis=-1, keepdims=False)  # (len_data, seq_len)
    art_ts = np.mean(art_ts, axis=-1, keepdims=False)  # (len_data, seq_len)
    tsne_ts = np.concatenate((ori_ts, art_ts), axis=0)   # (2 * len_data, seq_len)

    plt.figure(figsize=(16, 12))

    pca = PCA(n_components=2)
    pca.fit(ori_ts)
    pca_ori = pca.transform(ori_ts)
    pca_art = pca.transform(art_ts)

    plt.subplot(121)
    plt.grid()
    plot_scatter(pca_ori[:, 0], pca_ori[:, 1], color='b', alpha=0.1, label='Original')
    plot_scatter(pca_art[:, 0], pca_art[:, 1], color='r', alpha=0.1, label='Synthetic')
    plt.legend()
    plt.title(f'PCA plots for features averaged')
    plt.xlabel('x-pca')
    plt.ylabel('y-pca')

    tsne = TSNE(n_components=2)
    tsne_results = tsne.fit_transform(tsne_ts)

    plt.subplot(122)
    plt.grid()

    plot_scatter(tsne_results[:len_data, 0],
                 tsne_results[:len_data, 1],
                 color='b', alpha=0.1, label='Original')

    plot_scatter(tsne_results[len_data:, 0],
                 tsne_results[len_data:, 1],
                 color='r', alpha=0.1, label='Synthetic')

    plt.legend()
    plt.title(f't-SNE plots for features averaged')
    plt.xlabel('x-tsne')
    plt.ylabel('y-tsne')

    file_name = f'{args.experiment_name}_visualization.png'
    file_dir = os.path.join(args.pics_dir, file_name)
    plt.savefig(file_dir)


# 可视化传统AE
def plot_time_series_no_masks(args):
    """
        ori blue, art_mask red art_no_mask green
        原始数据是蓝色
        被掩盖后再生成的数据是红色
        没有被掩盖的在生成的数据是绿色
    """
    # 加载原始数据和后面生成数据
    ori_ts = np.load(args.ori_data_dir)   # (len_data, seq_len, z_dim)
    art_ts = np.load(args.art_data_dir)  # (len_data, seq_len, z_dim)
    # 选择要进行绘图的数据组数
    for i in range(args.samples_to_plot):
        ts_ori = ori_ts[i]  # (seq_len, z_dim)
        ts_art = art_ts[i]  # (seq_len, z_dim)

        subplots = [231, 232, 233, 234, 235, 236]
        # 定义画布大小
        plt.figure(figsize=(16, 12))

        # Plot max 6 features 最多绘制6个特征
        for k in range(min(args.z_dim, 6)):
            # 所有行的第k个特征
            ts_ori_k = ts_ori[:, k]  # (seq_len, )
            ts_art_k = ts_art[:, k]  # (seq_len, )
            # subplots中的数字表示row，col，index 比如231：row=2 col=3 index=6
            plt.subplot(subplots[k])
            # 是否显示网格线，默认为true
            plt.grid()
            plot_scatter(range(args.ts_size), ts_ori_k, color='b', label='Original')
            plot_scatter(range(args.ts_size), ts_art_k, color='g', label='Synthetic')
            plt.legend()
            plt.title(f'{args.columns[k]}')
        file_dir = os.path.join(args.pics_dir, f'sample{i}.png')
        plt.savefig(file_dir)


def plot_time_series_with_masks(args):
    """ori blue, art_mask red art_no_mask green"""
    ori_ts = np.load(args.ori_data_dir)   # (len_data, seq_len, z_dim)
    art_ts = np.load(args.art_data_dir)  # (len_data, seq_len, z_dim)
    masks = np.load(args.masks_dir)  # (len_data, seq_len)
    for i in range(args.samples_to_plot):
        ts_ori = ori_ts[i]  # (seq_len, z_dim)
        ts_art = art_ts[i]  # (seq_len, z_dim)
        mask = masks[i]  # (seq_len, )

        subplots = [231, 232, 233, 234, 235, 236]
        plt.figure(figsize=(16, 12))

        # Plot max 6 features
        for k in range(min(args.z_dim, 6)):
            ts_ori_k = ts_ori[:, k]  # (seq_len, )
            ts_art_k = ts_art[:, k]  # (seq_len, )
            ts_art_k_mask = ts_art_k[mask]
            ts_art_k_no_mask = ts_art_k[~mask]
            plt.subplot(subplots[k])
            plt.grid()
            plot_scatter(range(args.ts_size), ts_ori_k, color='b', label='Original')
            plot_scatter([j for j in range(args.ts_size) if mask[j]], ts_art_k_mask, color='r', label='Masked')
            plot_scatter([j for j in range(args.ts_size) if not mask[j]], ts_art_k_no_mask, color='g', label='Unmasked')
            plt.legend()
            plt.title(f'{args.columns[k]}')
        file_dir = os.path.join(args.pics_dir, f'sample{i}.png')
        plt.savefig(file_dir)


def get_length():
    osimples_length = np.loadtxt('data/ArgoLength_MAE.csv', delimiter=',', skiprows=1).tolist()
    simples_length = []
    for i in range(len(osimples_length)):
        simples_length.append(int(osimples_length[i]))
    return simples_length


def merge_data(sdata, simples_length):
    datas = []
    j, l, flag = 0, 0, 1
    for s in range(10):
        data = []
        simple_length = simples_length[s]
        # print('当前序列原始长度为{}'.format(simple_length))
        if s == 0:
            for j in range(sdata[s].shape[0]):
                data.append(sdata[s][j])
            j = j + 1
        dl = len(data)
        if flag == 0:
            j = j + 24
        while dl < simple_length:
            # print('当前遍历的序列号是{}'.format(j))
            for k in range(l, sdata[j].shape[0]):
                data.append(sdata[j][k])
                dl += 1
                if dl == simple_length:
                    if k < sdata[j].shape[0] - 1:
                        l = k + 1
                        # print('当前序列号是{}，有剩余，当前位置下一个位置是{}'.format(j, l))
                    else:
                        l = 0
                        flag = 0
                        # print('当前序列号是{}，无剩余，当前位置下一个位置是{}'.format(j, l))
                    break
            if dl < simple_length:
                l = 0
                j = j + 24
        data = np.array(data)
        datas.append(data)
    return datas


def get_merge_data(data):
    length = get_length()
    return merge_data(data, length)


def argo_plot_time_series_with_masks(args, save_path):
    """ori blue, art_mask red art_no_mask green"""
    ori_ts = np.load(args.ori_data_dir)  # (len_data, seq_len, z_dim)
    art_ts = np.load(args.art_data_dir)  # (len_data, seq_len, z_dim)
    masks = np.load(args.masks_dir)  # (len_data, seq_len)

    odata = get_merge_data(ori_ts)
    adata = get_merge_data(art_ts)
    ori_data = np.array(odata)
    art_data = np.array(adata)
    masks = np.array(get_merge_data(masks))
    masks = np.array(masks)

    for i in range(args.samples_to_plot):
        ts_ori = ori_data[i]  # (seq_len, z_dim)
        ts_art = art_data[i]  # (seq_len, z_dim)
        mask = masks[i]  # (seq_len, )

        subplots = [231, 232, 233, 234, 235, 236]
        plt.figure(figsize=(16, 12))
        depth = ts_ori[:, 0]

        # Plot max 4 features
        for k in range(1, min(args.z_dim, 4)):
            ts_ori_k = ts_ori[:, k]  # (seq_len, )
            ts_art_k = ts_art[:, k]  # (seq_len, )
            ts_art_k_mask = ts_art_k[mask]
            ts_art_k_no_mask = ts_art_k[~mask]

            art_mask = []
            art_no_mask = []
            for j in range(len(depth)):
                if mask[j]:
                    art_mask.append(j)
            for j in range(len(depth)):
                if not mask[j]:
                    art_no_mask.append(j)
            art_mask = np.array(depth[art_mask])
            art_no_mask = np.array(depth[art_no_mask])

            plt.subplot(subplots[k])
            plt.grid()
            plot_scatter(ts_ori_k, -depth, color='b', label='Original')
            plot_scatter(ts_art_k_mask, -art_mask, color='r', label='Masked')
            plot_scatter(ts_art_k_no_mask, -art_no_mask, color='g', label='Unmasked')
            plt.legend()
            plt.title(f'{args.columns[k]}')
        file_dir = os.path.join(save_path, f'sample{i}.png')
        plt.savefig(file_dir)
        plt.show()


def argo_plot_time_series_no_masks(args, save_path):
    """
        ori blue, art_mask red art_no_mask green
        原始数据是蓝色
        被掩盖后再生成的数据是红色
        没有被掩盖的在生成的数据是绿色
    """
    # 加载原始数据和后面生成数据
    ori_ts = np.load(args.ori_data_dir)   # (len_data, seq_len, z_dim)
    art_ts = np.load(args.art_data_dir)  # (len_data, seq_len, z_dim)

    odata = get_merge_data(ori_ts)
    adata = get_merge_data(art_ts)
    ori_data = np.array(odata)
    art_data = np.array(adata)

    # 选择要进行绘图的数据组数
    for i in range(args.samples_to_plot):
        ts_ori = ori_data[i]  # (seq_len, z_dim)
        ts_art = art_data[i]  # (seq_len, z_dim)

        subplots = [231, 232, 233, 234, 235, 236]
        # 定义画布大小
        plt.figure(figsize=(16, 12))
        depth = ts_ori[:, 0]

        # Plot max 6 features 最多绘制4个特征
        for k in range(min(args.z_dim, 4)):
            # 所有行的第k个特征
            ts_ori_k = ts_ori[:, k]  # (seq_len, )
            ts_art_k = ts_art[:, k]  # (seq_len, )
            # subplots中的数字表示row，col，index 比如231：row=2 col=3 index=6
            plt.subplot(subplots[k])
            # 是否显示网格线，默认为true
            plt.grid()
            plot_scatter(ts_ori_k, -depth, color='b', label='Original')
            plot_scatter(ts_art_k, -depth, color='g', label='Synthetic')
            plt.legend()
            plt.title(f'{args.columns[k]}')
        file_dir = os.path.join(save_path, f'sample{i}.png')
        plt.savefig(file_dir)
        plt.show()
