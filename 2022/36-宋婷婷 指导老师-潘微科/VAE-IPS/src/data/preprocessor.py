"""
Codes for preprocessing real-world datasets used in the experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
import codecs
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def preprocess_dataset(threshold: int = 4) -> Tuple:
    """Load and Preprocess datasets."""
    # load dataset.
    col = {0: 'user', 1: 'item', 2: 'rate'}
    with codecs.open(f'../..//data/train.txt', 'r', 'utf-8', errors='ignore') as f:
        data_train = pd.read_csv(f, delimiter=',', header=None)
        data_train.rename(columns=col, inplace=True)
    with codecs.open(f'../../data/test.txt', 'r', 'utf-8', errors='ignore') as f:
        data_test = pd.read_csv(f, delimiter=',', header=None)
        data_test.rename(columns=col, inplace=True)

    # print('data_train:\n', data_train)    # [311704 rows x 3 columns]: (u, i, rating)
    num_users, num_items = data_train.user.max() + 1, data_train.item.max() + 1    # num_users:15400, num_items:1000

    for _data in [data_train, data_test]:
        _data.user, _data.item = _data.user, _data.item
        # binalize rating.
        _data.rate[_data.rate < threshold] = 0
        _data.rate[_data.rate >= threshold] = 1

    # train-val-test, split
    train, test = data_train.values, data_test.values   # train:(311704, 3)  test:(54000, 3)
    train, val = train_test_split(train, test_size=0.1, random_state=12345)   # train:(280533, 3)  val:(31171, 3)

    # estimate pscore
    _, item_freq = np.unique(train[train[:, 2] == 1, 1], return_counts=True)
    pscore = (item_freq / item_freq.max()) ** 0.5   # size=1000

    # only positive data
    train_pos = train[train[:, 2] == 1, :2]    # (112406, 2)
    val = val[val[:, 2] == 1, :2]    # (12671, 2)

    # creating training data
    all_data = pd.DataFrame(np.zeros((num_users, num_items))).stack().reset_index()    # [15400000 rows x 3 columns]: (u, i, relevance)
    all_data = all_data.values[:, :2]   # (15400000, 2)
    unlabeled_data = np.array(list(set(map(tuple, all_data)) - set(map(tuple, train_pos))), dtype=int)    # (15287594, 2)
    train = np.r_[np.c_[train_pos, np.ones(train_pos.shape[0])], np.c_[unlabeled_data, np.zeros(unlabeled_data.shape[0])]]    # (15400000, 3)

    # save datasets
    path = Path(f'../../data/point')
    path.mkdir(parents=True, exist_ok=True)
    np.save(str(path / 'train.npy'), arr=train.astype(np.int))
    np.save(str(path / 'train_pos.npy'), arr=train_pos.astype(np.int))
    np.save(str(path / 'val.npy'), arr=val.astype(np.int))
    np.save(str(path / 'test.npy'), arr=test.astype(np.int))
    np.save(str(path / 'pscore.npy'), arr=pscore)
    np.save(str(path / 'item_freq.npy'), arr=item_freq)

if __name__ == '__main__':
    preprocess_dataset(4)