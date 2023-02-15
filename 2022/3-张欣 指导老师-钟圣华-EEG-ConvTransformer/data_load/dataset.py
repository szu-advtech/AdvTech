# encoding: utf-8
"""
 @author: Xin Zhang
 @contact: 2250271011@email.szu.edu.cn
 @time: 2022/11/9 15:22
 @name: 
 @desc:
"""
import torch
from torch.utils.data.dataset import Dataset
from data_load.serialize import file_scanf
import pickle


class EEGImagesDataset(Dataset):
    """EEGLearn Images Dataset from EEG."""

    def __init__(self, path):
        self.filepaths = file_scanf(path, endswith='.pkl')

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]

        with open(filepath, 'rb') as f:
            x = pickle.load(f)  # [t, w, h]
            y = int(pickle.load(f))

            y = y - 1
            assert 0 <= y <= 5

        return torch.tensor(x, dtype=torch.float).permute(1, 2, 0).unsqueeze(0), torch.tensor(y, dtype=torch.long)


