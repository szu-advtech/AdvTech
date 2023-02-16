import numpy as np

from utils.dataset import DatasetSplit


def iid(dataset, start_index, end_index):
    # num_items = int(len(dataset) / num_users)
    all_idxs = [i for i in range(len(dataset))]
    return DatasetSplit(dataset, all_idxs[start_index: end_index])


def non_iid(dataset, user_id, num_users):
    num_shards = num_users * 2
    num_imgs = int(len(dataset) / num_shards)
    labels = dataset.train_labels.numpy()
    idxs = np.arange(len(dataset))
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    result = np.array([], dtype='int64')
    result = np.concatenate((result, idxs[user_id * num_imgs: (user_id + 1) * num_imgs]), axis=0)
    result = np.concatenate((result, idxs[(num_shards - user_id - 1) * num_imgs: (num_shards - user_id) * num_imgs]), axis=0)
    return DatasetSplit(dataset, result)
