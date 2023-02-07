
import os
import pickle
from torch.utils.data import Dataset
from collections import defaultdict

class MetaDataset(Dataset):

    def __init__(self, dataset, labels_to_indices=None, indices_to_labels=None, users_to_indices=None, indices_to_users=None):

        if not isinstance(dataset, Dataset):
            raise TypeError("MetaDataset only accepts a torch dataset as input")

        self.dataset = dataset

        if hasattr(dataset, '_bookkeeping_path'):
            self.load_bookkeeping(dataset._bookkeeping_path)
        else:
            self.create_bookkeeping(
                labels_to_indices=labels_to_indices,
                indices_to_labels=indices_to_labels,
                users_to_indices=users_to_indices,
                indices_to_users=indices_to_users,
            )

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def create_bookkeeping(self, labels_to_indices=None, indices_to_labels=None, users_to_indices=None, indices_to_users=None, tasks_to_indices=None):
        """
        Iterates over the entire dataset and creates a map of target to indices.
        遍历整个数据集并创建目标到索引的映射。
        返回:一个字典，key为标签，value为索引列表。
        Returns: A dict with key as the label and value as list of indices.
        """

        assert hasattr(self.dataset, '__getitem__'), \
            'Requires iterable-style dataset.'

        # Bootstrap from arguments
        if labels_to_indices is not None:
            indices_to_labels = {
                idx: label
                for label, indices in labels_to_indices.items()
                for idx in indices
            }
        elif indices_to_labels is not None:
            labels_to_indices = defaultdict(list)
            for idx, label in indices_to_labels.items():
                labels_to_indices[label].append(idx)
        else:  # Create from scratch
            labels_to_indices = defaultdict(list)
            indices_to_labels = defaultdict(int)
            users_to_indices = defaultdict(list)
            indices_to_users = defaultdict(int)
            tasks_to_indices = defaultdict(list)
            indices_to_tasks = defaultdict(int)
            for i in range(len(self.dataset)):
                try:
                    label = self.dataset[i][1]
                    # if label is a Tensor, then take get the scalar value
                    if hasattr(label, 'item'):
                        label = self.dataset[i][1].item()
                except ValueError as e:
                    raise ValueError(
                        'Requires scalar labels. \n' + str(e))

                labels_to_indices[label].append(i)
                indices_to_labels[i] = label
                try:
                    user = self.dataset[i][2]
                    # if label is a Tensor, then take get the scalar value
                    if hasattr(user, 'item'):
                        user = self.dataset[i][2].item()
                except ValueError as e:
                    raise ValueError(
                        'Requires scalar labels. \n' + str(e))
                users_to_indices[user].append(i)
                indices_to_users[i] = user
                try:
                    task = self.dataset[i][3]
                    # if label is a Tensor, then take get the scalar value
                    if hasattr(label, 'item'):
                        task = self.dataset[i][3].item()
                except ValueError as e:
                    raise ValueError(
                        'Requires scalar labels. \n' + str(e))
                tasks_to_indices[task].append(i)
                indices_to_tasks[i] = task
            
        self.labels_to_indices = labels_to_indices
        self.indices_to_labels = indices_to_labels
        self.labels = list(self.labels_to_indices.keys())
        
        self.indices_to_users = indices_to_users
        self.users_to_indices = users_to_indices
        self.users = list(self.users_to_indices.keys())

        self.tasks_to_indices = tasks_to_indices
        self.indices_to_tasks = indices_to_tasks
        self.tasks = list(self.tasks_to_indices.keys())

        self._bookkeeping = {
            'labels_to_indices': self.labels_to_indices,
            'indices_to_labels': self.indices_to_labels,
            'labels': self.labels
        }

    def load_bookkeeping(self, path):
        if not os.path.exists(path):
            self.create_bookkeeping()
            self.serialize_bookkeeping(path)
        else:
            with open(path, 'rb') as f:
                self._bookkeeping = pickle.load(f)
            self.labels_to_indices = self._bookkeeping['labels_to_indices']
            self.indices_to_labels = self._bookkeeping['indices_to_labels']
            self.labels = self._bookkeeping['labels']

    def serialize_bookkeeping(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self._bookkeeping, f, protocol=-1)