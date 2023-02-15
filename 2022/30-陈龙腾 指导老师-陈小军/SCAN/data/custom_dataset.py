"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.utils import rand_bbox
import copy
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, Sampler
import itertools
import matplotlib.pyplot as plt
from PIL import Image


""" 
    AugmentedDataset
    Returns an image together with an augmentation.
"""
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset

        if isinstance(transform, dict):
            self.image_transform = transform['standard'] #
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset.__getitem__(index)
        image = x['image']
        sample = {}
        sample['image'] = self.image_transform(image)
        sample['image_augmented'] = self.augmentation_transform(image)
        sample['index'] = index
        sample['target'] = x['target']
        return sample


""" 
    NeighborsDataset
    Returns an image with one of its neighbors.
"""


class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, num_neighbors=None):
        super(NeighborsDataset, self).__init__()
        # self.R = R
        # self.batch = batch
        transform = dataset.transform
        
        if isinstance(transform, dict):
            self.anchor_transform = transform['standard']
            self.neighbor_transform = transform['augment']
            self.mix_transform = transform['standard']

        else:
            self.anchor_transform = transform
            self.neighbor_transform = transform
            self.mix_transform = transform

        dataset.transform = None
        self.dataset = dataset
        self.indices = indices # Nearest neighbor indices (np.array  [len(dataset) x k])
        if num_neighbors is not None:
            self.indices = self.indices[:, :num_neighbors+1]
        assert(self.indices.shape[0] == len(self.dataset)) # 断定这个程序执行之后或者之前会有这样的结果,如果不是,那就扔出一个错误



    def __len__(self):
        return len(self.dataset)



    def __getitem__(self, index):

        output = {}
        anchor = self.dataset.__getitem__(index)

        # 保持原样本和邻居在特征空间一致性
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        output['image'] = self.anchor_transform(anchor['image'])
        output['image_augmented'] = self.neighbor_transform(anchor['image'])
        output['anchor'] = self.anchor_transform(anchor['image'])
        output['neighbor'] = self.neighbor_transform(neighbor['image'])
        output['anchor_neighbors_indices'] = torch.from_numpy(self.indices[index])
        output['neighbor_neighbors_indices'] = torch.from_numpy(self.indices[neighbor_index])
        output['target'] = anchor['target']
        output['index'] = index
        output['rd'] = 1
        
        return output
