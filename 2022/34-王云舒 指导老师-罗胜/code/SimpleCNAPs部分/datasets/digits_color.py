  #!/usr/bin/env python3

from __future__ import print_function
import os
import pickle
import numpy as np
import torch
import torch.utils.data as data
from skimage import io
import skimage
from skimage import transform as transform1
from path_index import *

def index_classes(items):
    idx = {}
    for i in items:
        if (i not in idx):
            idx[i] = len(idx)
    return idx


class digits_color(data.Dataset):

    def __init__(self,
                 root,
                 mode='train',
                 transform=None,
                 target_transform=None,
                 download=False):
        super(digits_color, self).__init__()
        self.root = os.path.expanduser(root)
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        pkl_dir = ""
        if mode == 'train':
            if pretrainORmetatrainFLAG == "pretrain":
                pkl_dir = pretrain_traindataset
            else:
                pkl_dir = metatrain_traindataset
        elif mode == 'valid':
            if pretrainORmetatrainFLAG == "pretrain":
                pkl_dir = pretrain_validdataset
            else:
                pkl_dir = metatrain_validdataset
        else:
            if pretrainORmetatrainFLAG == "pretrain":
                pkl_dir = pretrain_testdataset
            else:
                pkl_dir = metatrain_testdataset
        
        pkl_file = open(pkl_dir, 'rb')
        data1 = pickle.load(pkl_file)
        pkl_file.close()
        print("load finished, successfully loaded "+self.mode)
        i = len(data1['image_data'])
        print("dataset size = ",i)
        # print("dataset size =")
        # print(i)
        # file = open('gesture'+'_'+self.mode+'.pkl', 'wb')
        # pickle.dump(data1, file)
        # file.close()
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
        self.task_names = {}
        for task_name, idxs in self.data['img_task'].items():
            for idx in idxs:
                self.task[idx] = self.task_idx[task_name]
                self.task_names[self.task_idx[task_name]] = task_name

    def __getitem__(self, idx):
        data = self.x[idx]
        if self.transform:
            data = self.transform(data)
        return data, self.y[idx], self.z[idx], self.task[idx]

    def __len__(self):
        return len(self.x)

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'mini-imagenet-cache-' + self.mode + '.pkl'))


if __name__ == '__main__':
    mi = digits_color(root='./data', download=True)
    __import__('pdb').set_trace()
