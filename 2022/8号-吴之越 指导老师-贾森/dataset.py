import torch
from torch.utils.data import Dataset
import numpy as np
from utils import *
import os
from torch.nn import functional as f
import torchvision.transforms as t


class dataset(Dataset):
    def __init__(self, is_downsample, is_train, scale_factor = 4):
        """
        is_train: Means training or validation
        is_downsample: Means Downsampling or not
        """
        if is_train:
            self.file_root_mat_path = './datasets/chikusei/train/mat'
            self.file_root_rgb_path = './datasets/chikusei/train/rgb'
            self.transform = t.Compose([
                t.ToTensor(),
                t.RandomRotation(0.5),
                t.RandomHorizontalFlip(0.5)
            ])
        else:
            self.file_root_mat_path = './datasets/chikusei/validation/mat'
            self.file_root_rgb_path = './datasets/chikusei/validation/rgb'
            self.transform = t.ToTensor()

        self.train_img_mat_list = os.listdir(self.file_root_mat_path)
        self.train_img_rgb_list = os.listdir(self.file_root_rgb_path)
        self.scale_factor = scale_factor
        self.is_downsample = is_downsample
        self.is_train = is_train

    def __getitem__(self, index):
        assert len(self.train_img_rgb_list) == len(self.train_img_mat_list)

        img_mat_path = os.path.join(self.file_root_mat_path, self.train_img_mat_list[index])
        img_rgb_path = os.path.join(self.file_root_rgb_path, self.train_img_rgb_list[index])

        img_mat_y = np.load(img_mat_path)
        img_mat_y = self.transform(img_mat_y)
        img_mat_y = (img_mat_y - img_mat_y.min()) / (img_mat_y.max() - img_mat_y.min())

        img_rgb_y = np.load(img_rgb_path)
        img_rgb_y = self.transform(img_rgb_y)
        img_rgb_y = (img_rgb_y - img_rgb_y.min()) / (img_rgb_y.max() - img_rgb_y.min())

        if self.is_downsample:
            # 下采样
            # img_x = img_y[:, int(self.scale_factor // 2) - 1::self.scale_factor, int(self.scale_factor // 2) - 1::self.scale_factor]
            img_mat_y = img_mat_y.reshape(1, img_mat_y.shape[0], img_mat_y.shape[1], img_mat_y.shape[2])
            img_mat_x = f.interpolate(img_mat_y, scale_factor=0.25, mode='bicubic').squeeze()
            img_mat_y = img_mat_y.squeeze(dim=0)

        else:
            img_mat_x = img_mat_y


        img_rgb_x = img_rgb_y
        if self.is_train:
            img_mat_x.requires_grad, img_mat_y.requires_grad = True, True
            img_rgb_x.requires_grad, img_rgb_y.requires_grad = True, True

        else:
            pass

        return (img_mat_x, img_rgb_x), (img_mat_y, img_rgb_y)

    def __len__(self):
        return len(self.train_img_mat_list)
