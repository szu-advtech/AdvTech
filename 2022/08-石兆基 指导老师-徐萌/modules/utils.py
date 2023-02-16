"""
ExtraMAE所有实用函数
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

import torch

"""utils.py includes
    1. Args Loading
    2. Data Related
    3. Model Related"""

"""Args Loading"""


# 加载数据相关信息
def load_arguments(home):
    # Find the config for experiments  找到实验配置
    # argparse表示读取命令行中的参数
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_dir', default='argo_config.json')
    # vars表示返回对象的属性和属性值的字典对象
    args_dict = vars(parser.parse_args())

    # Load the stock_config.json
    # 从命令行中获取的数据目录存入config-dir
    config_dir = args_dict['config_dir']

    # 读入相应数据的config文件
    with open(config_dir, 'r') as f:
        # 读取的文件是一个字典类型
        config_dict = json.load(fp=f)

    config_dict['home'] = home

    # 将当前工作目录与获取的数据相关文件的目录放在一个总字典中
    total_dict = {**config_dict, **args_dict}

    # Maintain dirs  维护目录
    storage_dir = os.path.join(home, 'storage')
    total_dict['storage_dir'] = storage_dir
    # 如果storage_dir这个文件不存在，则创建该文件
    if not os.path.isdir(storage_dir):
        os.mkdir(storage_dir)

    # 在storage目录下创建一下文件夹
    experiment_dir = os.path.join(storage_dir, config_dict['experiment_name'])
    model_dir = os.path.join(experiment_dir, 'model')
    pics_dir = os.path.join(experiment_dir, 'pics')
    synthesis_dir = os.path.join(experiment_dir, 'synthesis')

    # 把上述创建文件夹的地址存入总字典中，方便后续使用
    total_dict['experiment_dir'] = experiment_dir
    total_dict['model_dir'] = model_dir
    total_dict['pics_dir'] = pics_dir
    total_dict['synthesis_dir'] = synthesis_dir

    print(f'experiment_dir is {experiment_dir}')
    if not os.path.isdir(experiment_dir):
        os.mkdir(experiment_dir)
        # create main sub folders for the experiments
        os.mkdir(model_dir)
        os.mkdir(pics_dir)
        os.mkdir(synthesis_dir)

    # Maintain dirs for Data  维护数据目录
    # 获取数据所在目录
    datasets_dir = os.path.join(home, 'data')
    total_dict['datasets_dir'] = datasets_dir
    # dirs for specific dataset
    # 获取特定数据集的地址
    total_dict['stock_dir'] = os.path.join(datasets_dir, 'stock_data.csv')
    total_dict['energy_dir'] = os.path.join(datasets_dir, 'energy_data.csv')
    # total_dict['xbt_dir'] = os.path.join(datasets_dir, 'xbt_data_15.csv')
    total_dict['argo_dir'] = os.path.join(datasets_dir, 'Argo_MAE.csv')

    art_data_dir = os.path.join(model_dir, 'art_data.npy')
    ori_data_dir = os.path.join(model_dir, 'ori_data.npy')
    masks_dir = os.path.join(model_dir, 'masks.npy')
    total_dict['art_data_dir'] = art_data_dir
    # 待加载数据并切片后将原始数据存入该目录下
    total_dict['ori_data_dir'] = ori_data_dir
    total_dict['masks_dir'] = masks_dir

    args = argparse.Namespace(**total_dict)
    print(args)
    return args


"""Data Related"""


# 最大最小归一化数据
def min_max_scalar(data):
    """Min-Max Normalizer.   最大最小归一化

    Args:
      - data: raw data  原始数据

    Returns:
      - norm_data: normalized data   标准化数据
      - min_val: minimum values (for renormalization)  最小值
      - max_val: maximum values (for renormalization)  最大值
    """
    # np.min(axis=0/1), 0意味返回矩阵中每列最小的值， 1意味着返回矩阵中每行最小的值
    min_val = np.min(np.min(data, axis=0), axis=0)  # (z_dim, ) min for each feature
    data = data - min_val

    max_val = np.max(np.max(data, axis=0), axis=0)  # (z_dim, ) max for each feature
    norm_data = data / (max_val + 1e-7)

    return norm_data, min_val, max_val


# 正弦数据生成函数
def sine_data_generation(num_samples, seq_len, z_dim):
    """
    Sine data generation
       Remark: no args.min/max/var for sine_data
               no normalization   无标准化
               no renormalization  无重整化
    Args:
        - num_samples: the number of samples  样本数量
        - seq_len: the sequence length of the time-series   序列片段长度
        - dim: feature dimensions  特征尺寸
    Returns:
        - data: generated data
    """
    sine_data = list()
    for i in range(num_samples):
        single_sample = list()
        for k in range(z_dim):
            # Randomly drawn frequency and phase for each feature (column)
            # 均匀分布，low为下界，high为上界
            freq = np.random.uniform(low=0, high=0.1)
            phase = np.random.uniform(low=0, high=0.1)
            sine_feature = [np.sin(freq * j + phase) for j in range(seq_len)]
            single_sample.append(sine_feature)
        # 将生成的数据转为array格式后再转置，transpose为转置
        # 这样得到的就是以z_dim为特征数，seq_len为序列片段长度的一个序列片段数据
        single_sample = np.transpose(np.asarray(single_sample))  # (seq_len, z_dim)
        single_sample = (single_sample + 1) * 0.5
        # Stack the generated data
        sine_data.append(single_sample)
    sine_data = np.array(sine_data)  # (num_sample, seq_len, z_dim)
    return sine_data


# 对原始数据切片
def sliding_window(args, ori_data):
    """ Slicing the ori_data by sliding window
        Args:
            args
            ori_data (len(csv), z_dim)
        Returns:
            ori_data (:, seq_len, z_dim)"""
    # Flipping the data to make chronological data
    # 翻转数据（将数据顺序反过来），制作序列数据
    if args.continues == 0:
        ori_data = ori_data[::-1]  # (len(csv), z_dim)

    # Make (len(ori_data), z_dim) into (num_samples, seq_len, z_dim)
    samples = []

    # ts_size为序列片段数量
    for i in range(len(ori_data)-args.ts_size):
        single_sample = ori_data[i:i + args.ts_size]  # (seq_len, z_dim)
        samples.append(single_sample)
    samples = np.array(samples)  # (bs, seq_len, z_dim)

    # shuffle类似于洗牌，返回一个重新排序的随机序列
    if args.continues == 0:
        np.random.shuffle(samples)  # Make it more like i.i.d.
    return samples


# z_dim表示数据的特征数量
# 加载数据
def load_data(args):
    """
    Load and preprocess rea-world datasets and record necessary statistics
    Args:
        - data_name: stock or energy  数据名：股票或能源
        - seq_len: sequence length    序列长度
    Returns:
        - data: preprocessed data 返回与处理的数据
    """
    # 判断数据是否为股票、能源或正弦数据
    assert args.data_name in ['stock', 'energy', 'sine', 'argo']
    ori_data = None

    """
    delimiter：分隔符
    skiprows=NUM：跳过数据文件前NUM行 
    """
    if args.data_name == 'stock':
        # 跳过第一行，将原始数据按照“，”分隔开
        ori_data = np.loadtxt(args.stock_dir, delimiter=',', skiprows=1)
        ori_data = sliding_window(args, ori_data)
        # pd.DataFrama.columns为返回数据的列名
        args.columns = pd.read_csv(args.stock_dir).columns
    elif args.data_name == 'energy':
        ori_data = np.loadtxt(args.energy_dir, delimiter=',', skiprows=1)
        ori_data = sliding_window(args, ori_data)
        args.columns = pd.read_csv(args.energy_dir).columns
        print(ori_data.shape)
    elif args.data_name == 'sine':
        ori_data = sine_data_generation(num_samples=10000, seq_len=args.ts_size, z_dim=args.z_dim)
        args.columns = [f'feature{i}' for i in range(args.z_dim)]
    elif args.data_name == 'argo':
        ori_data = np.loadtxt(args.argo_dir, delimiter=',', skiprows=1)
        ori_data = sliding_window(args, ori_data)
        args.columns = pd.read_csv(args.argo_dir).columns

    # saving the processed data for work under args.working_dir
    np.save(args.ori_data_dir, ori_data)
    return ori_data


# 批量获取数据
def get_batch(args, data):
    # permutation表示将数据随机排列
    idx = np.random.permutation(len(data))
    idx = idx[:args.batch_size]
    # 随机选取batch_size个数据作为初始训练数据
    data_mini = data[idx, ...]  # (bs, seq_len, z_dim)
    return data_mini


# 训练、测试集划分
def train_test_split(args, data):
    # Split ori_data
    idx = np.random.permutation(len(data))
    train_idx = idx[:int(args.train_test_ratio * len(data))]
    test_idx = idx[int(args.train_test_ratio * len(data)):]
    train_data = data[train_idx, ...]
    test_data = data[test_idx, ...]
    return train_data, test_data


"""Model Related"""

# 保存模型参数
def save_model(args, model):
    # 创建一个下面的目录文件
    file_dir = os.path.join(args.model_dir, 'model.pth')
    # 将相关参数保存到该目录文件
    torch.save(model.state_dict(), file_dir)


# 保存评估结果
def save_metrics_results(args, results):
    file_dir = os.path.join(args.model_dir, 'metrics_results.npy')
    np.save(file_dir, results)


# 保存数据相关信息
def save_args(args):
    file_dir = os.path.join(args.model_dir, 'args_dict.npy')
    np.save(file_dir, args.__dict__)


# 加载模型
def load_model(args, model):
    model_dir = args.model_dir
    file_dir = os.path.join(model_dir, 'model.pth')

    model_state_dict = torch.load(file_dir)
    model.load_state_dict({f'model.{k}': v for k, v in model_state_dict.items()})
    return model


# 加载字典文件
def load_dict_npy(file_path):
    file = np.load(file_path, allow_pickle=True)
    return file


"""For TimeGAN metrics"""


# 在计算discrimination分数训练模型时要划分训练/测试集
def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """
    Divide train and test data for both original and synthetic data.
    分割原始数据和合成数据的训练和测试数据。
    Args:
        - data_x: original data  原始数据
        - data_x_hat: generated data  原始时间
        - data_t: original time  生成数据
        - data_t_hat: generated time  生成时间
        - train_rate: ratio of training data from the original data  训练数据的比率
    """
    # Divide train/test index (original data) 对原始数据划分训练/测试集 比例为8：2
    # permute the indexies and split the first 0.8 percent to be training data
    no = len(data_x)
    # 随机生成一些数据
    idx = np.random.permutation(no)
    # 取前80%的随机数据为训练数据的下标，后20%的随机数据数据为测试数据的下标
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    # 根据下边取对应的数据和时间信息
    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data) 对生成数据划分训练/测试集 比例为8：2
    # Repeat it again for the synthetic data
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[:int(no * train_rate)]
    test_idx = idx[int(no * train_rate):]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return train_x, train_x_hat, test_x, test_x_hat, train_t, train_t_hat, test_t, test_t_hat


# 在计算discrimination分数训练模型时需要批量产生数据
def batch_generator(data, time, batch_size):
    """
    Mini-batch generator. Slice the original data to the size a batch.
    获取批量数据

    Args:
        - data: time-series data
        - time: time series length for each sample
        - batch_size: the number of samples in each batch  批量大小

    Returns:
        - X_mb: time-series data in each batch (bs, seq_len, dim)  批量数据
        - T_mb: time series length of samples in that batch (bs, len of the sample) 批量数据的长度
        """
    # randomly select a batch of idx
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    # picked the selected samples and their corresponding series length
    X_mb = list(data[i] for i in train_idx)
    T_mb = list(time[i] for i in train_idx)

    return X_mb, T_mb


# 序列长度信息获取
def extract_time(data):
    """
    Returns Maximum sequence length and each sequence length.
    返回最大序列长度和每个序列的长度

    Args:
    - data: original data (no, seq_len, dim)
    数据：原始数据（数据量，序列长度，特征维数）

    Returns:
    - time: a list for each sequence length
    - max_seq_len: maximum sequence length
    time：每个序列长度的列表
    max_seq_len：最大序列长度
    """
    time = list()
    max_seq_len = 0
    # 例：stock数据，每段长度都相同=24，所以最大也是24
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))
    return time, max_seq_len


def extract_factors(n):
    if (n == 0) or (n == 1):
        return [n]

    factor_list = []
    i = 2
    while i < n:
        if n % i == 0:
            factor_list.append(i)
        i += 1

    return factor_list
