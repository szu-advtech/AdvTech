import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import h5py
import numpy as np


def Normalize(data):
    mins, maxs = np.min(data), np.max(data)
    img_nor = (data - mins) / (maxs  - mins )
    return img_nor

class DeepLesionImageDataset(Dataset):
    def __init__(self, h5_list, h5_dir, transform=None, target_transform=None):
        self.h5File_list = pd.read_table(os.path.join(h5_dir, h5_list),sep=' ',header=None)
        self.h5_dir = h5_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return self.h5File_list.shape[0]

    def __getitem__(self, idx):

        ct_id = int(self.h5File_list.iloc[idx, 1])
        datafilepath = os.path.join(self.h5_dir, self.h5File_list.iloc[idx, 0])
        data = h5py.File(datafilepath)
        X_ldma = data['ma_CT'][ct_id]
        X_gt = data['gt_CT'][()]

        M = h5py.File(os.path.join(self.h5_dir, 'mask.h5'))['mask'][ct_id]

        X_ldma = np.expand_dims(X_ldma, axis=0)
        X_gt = np.expand_dims(X_gt, axis=0)
        M = np.expand_dims(M, axis=0)


        return torch.tensor(X_ldma), torch.tensor(X_gt), torch.tensor(M)


class testDeepLesionImageDataset(Dataset):
    def __init__(self, h5_list, h5_dir, transform=None, target_transform=None):
        self.h5File_list = pd.read_table(os.path.join(h5_dir, h5_list),sep=' ',header=None)
        self.h5_dir = h5_dir
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return self.h5File_list.shape[0]

    def __getitem__(self, idx):

        ct_id = int(self.h5File_list.iloc[idx, 1])
        datafilepath = os.path.join(self.h5_dir, self.h5File_list.iloc[idx, 0])
        data = h5py.File(datafilepath)
        X_ldma = data['ma_CT'][ct_id]
        X_gt = data['gt_CT'][()]

        M = h5py.File(os.path.join(self.h5_dir, 'mask.h5'))['mask'][ct_id]

        X_ldma = np.expand_dims(X_ldma, axis=0)
        X_gt = np.expand_dims(X_gt, axis=0)
        M = np.expand_dims(M, axis=0)


        return torch.tensor(X_ldma), torch.tensor(X_gt), torch.tensor(M),torch.tensor(idx)