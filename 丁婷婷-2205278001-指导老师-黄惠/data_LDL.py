# This code is written by Jingyuan Yang @ XD

"""Emotion_LDL Dataset class"""

from __future__ import absolute_import
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import autograd

class Emotion_LDL(data.Dataset):

    def __init__(self, csv_file, root_dir, transform):
        self.annotations = pd.read_csv(csv_file, header=None) # None: no header 0: 1st row is header
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = str(self.root_dir) + str(self.annotations.iloc[idx, 0]).split('.')[0] + '.jpg' # 0 1
        image = Image.open(img_name)
        image = image.convert("RGB")
        image = self.transform(image)
        dist_emo = self.annotations.iloc[idx, 1:9].values.astype("float32")
        dist_emo = dist_emo / dist_emo.sum() # norm
        # NO softmax, that will bring a distribution to a single label problem

        sample = {'img_id': str(self.annotations.iloc[idx, 0]).split('.')[0] + '.jpg',
                  'image': image, 'dist_emo': dist_emo}

        return sample


def softmax(x):
    x_exp = np.exp(x)
    #如果是列向量，则axis=0
    x_sum = np.sum(x_exp, axis = 0, keepdims = True)
    s = x_exp / x_sum
    return s