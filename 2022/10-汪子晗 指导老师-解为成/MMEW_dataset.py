import numpy as np
from torchvision import transforms
import torch
import torch.utils.data as data
import os
import re
import cv2
import pandas as pd

def make_dataset(onset,apex,emo):
    return [(onset[i], apex[i], emo[i]) for i in range(len(onset))]

class MMEW(data.Dataset):
    def __init__(self,root_path,train,num_loso,transform=None,basic_aug=False,transform_norm=None):
        self.root_path = root_path #/NAS_REMOTE/wangzihan/MMEW
        self.train = train
        self.transform = transform
        self.transform_norm = transform_norm
        self.basic_aug = basic_aug
        self.subject_out_idx = num_loso

        self.img_path = open(os.path.join('/NAS_REMOTE/wangzihan', 'list8/MMEW_onset_img_path.txt')).readlines()
        self.idx = []

        for i in range(len(self.img_path)):
            num = self.img_path[i].split('/')[1][1:3]
            num = int(num)
            if num == self.subject_out_idx:
                self.idx.append(i)
        if len(self.idx):
            start = self.idx[0]
            end = self.idx[-1]
        else:
            end,start = -1,0
        self.onset = open(os.path.join(self.root_path, 'list8', 'MMEW_onset_img_path.txt')).readlines()
        self.apex = open(os.path.join(self.root_path, 'list8', 'MMEW_apex_img_path.txt')).readlines()
        self.emotion = np.loadtxt(os.path.join(self.root_path, 'list8', 'MMEW_emo_label.txt'))

        if self.train:
            train_onset = self.onset[:start] + self.onset[end+1:]
            train_apex = self.apex[:start] + self.apex[end+1:]
            train_emo = np.delete(self.emotion, self.idx, 0)
            self.data_list = make_dataset(train_onset, train_apex, train_emo)
        else:
            test_onset = self.onset[start:end+1]
            test_apex = self.apex[start:end+1]
            test_emo = self.emotion[start:end+1]
            self.data_list = make_dataset(test_onset, test_apex, test_emo)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, item):
        onset, apex, emo = self.data_list[item]
        onset = '/NAS_REMOTE/wangzihan/MMEW/' + onset.strip()
        apex = '/NAS_REMOTE/wangzihan/MMEW/' + apex.strip()
        img_onset = cv2.imread(onset)
        img_apex = cv2.imread(apex)

        if self.transform is not None:
            img_onset = self.transform(img_onset)
            img_apex = self.transform(img_apex)
            all = torch.cat((img_onset, img_apex), dim=0)
            if self.transform_norm is not None and self.train:
                all = self.transform_norm(all)
            img_onset = all[0:3,:,:]
            img_apex = all[3:6,:,:]

        return img_onset, img_apex, int(emo)
