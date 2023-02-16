#!/usr/bin/env python3

from __future__ import print_function

import os

import numpy as np
import torch
import torch.utils.data as data

def index_classes(items):
    idx = {}
    for i in items:
        if (i not in idx):
            idx[i] = len(idx)
    return idx


class digits_set(data.Dataset):
    """
    [[Source]](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/mini_imagenet.py)

    **Description**

    The *mini*-ImageNet dataset was originally introduced by Vinyals et al., 2016.


    It consists of 60'000 colour images of sizes 84x84 pixels.
    The dataset is divided in 3 splits of 64 training, 16 validation, and 20 testing classes each containing 600 examples.
    The classes are sampled from the ImageNet dataset, and we use the splits from Ravi & Larochelle, 2017.

    **References**

    1. Vinyals et al. 2016. “Matching Networks for One Shot Learning.” NeurIPS.
    2. Ravi and Larochelle. 2017. “Optimization as a Model for Few-Shot Learning.” ICLR.

    **Arguments**

    * **root** (str) - Path to download the data.
    * **mode** (str, *optional*, default='train') - Which split to use.
        Must be 'train', 'validation', or 'test'.
    * **transform** (Transform, *optional*, default=None) - Input pre-processing.
    * **target_transform** (Transform, *optional*, default=None) - Target pre-processing.

    **Example**

    ~~~python
    train_dataset = l2l.vision.datasets.MiniImagenet(root='./data', mode='train')
    train_dataset = l2l.data.MetaDataset(train_dataset)
    train_generator = l2l.data.TaskGenerator(dataset=train_dataset, ways=ways)
    ~~~

    """

    def __init__(self,
                 root=None,
                 mode='train',
                 transform=None,
                 target_transform=None,
                 data1=None):
        super(digits_set, self).__init__()
        #self.root = os.path.expanduser(root)
        #if not os.path.exists(self.root):
        #    os.mkdir(self.root)
        self.transform = transform
        self.target_transform = target_transform
        #self.mode = mode
        #print("load finished, successfully loaded "+self.mode)
        i = len(data1['image_data'])
        print("dataset size = ",i)
        self.data = data1
        self.size = i

        self.x = torch.from_numpy(self.data["image_data"]).permute(0, 3, 1, 2).float()
        self.y = np.ones(len(self.x),dtype=int)
        self.z = np.ones(len(self.x),dtype=int)
        self.task = np.ones(len(self.x), dtype=int)

        # TODO Remove index_classes from here
        self.class_idx = index_classes(self.data['class_dict'].keys())
        for class_name, idxs in self.data['class_dict'].items():
            for idx in idxs:
                self.y[idx] = self.class_idx[class_name]

        self.user_idx = index_classes(self.data['img_user'].keys())
        for user_name, idxs in self.data['img_user'].items():
            for idx in idxs:
                self.z[idx] = self.user_idx[user_name]
        self.task_idx = index_classes(self.data['img_task'].keys())
        for task_name, idxs in self.data['img_task'].items():
            for idx in idxs:
                self.task[idx] = self.task_idx[task_name]

    def __getitem__(self, idx):
        data = self.x[idx]
        if self.transform:
            data = self.transform(data)
        return data, self.y[idx], self.z[idx],self.task[idx]

    def __len__(self):
        return len(self.x)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'mini-imagenet-cache-' + self.mode + '.pkl'))