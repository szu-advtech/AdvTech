from PIL import Image
import torchfile
from torch.utils.data import DataLoader, TensorDataset, random_split, ConcatDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
import torch
import torch.nn as nn
import os
import sys
import torchvision.utils as vutils
import numpy as np
import cv2
import torch.nn.init as init
import torch.utils.data as data
import random
import xlrd
import math
from skimage.exposure import match_histograms
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import ImageEnhance
from utils.utils import im_convert
from utils.data_utils import colorful_spectrum_mix, fourier_transform, save_image
from config import default_config


root = "../Dataset_BUSI_with_GT/"
prefix = "../Dataset_BUSI_with_GT/label/"

def make_dataset(dir):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            images.append(path)
    return images

def default_loader(path):
    return np.load(path)['arr_0']

def gt_cut(source_img, target_img, mask):
    source_img = np.array(source_img)
    target_img = np.array(target_img)
    mask = np.array(mask, dtype = bool)
    aug_img = source_img * mask + target_img * (~mask)
    return aug_img

def fourier_augmentation(img, tar_img, mode, alpha):
    # transfer image from PIL to numpy
    img = np.array(img)
    tar_img = np.array(tar_img)
    img = img[:,:,np.newaxis]
    tar_img = tar_img[:,:,np.newaxis]

    # the mode comes from the paper "A Fourier-based Framework for Domain Generalization"
    if mode == 'AS':
        # print("using AS mode")
        aug_img, aug_tar_img = fourier_transform(img, tar_img, L=0.02, i=0.8)  #0.01  1
    elif mode == 'AM':
        # print("using AM mode")
        aug_img, aug_tar_img = colorful_spectrum_mix(img, tar_img, alpha=alpha)
    else:
        print("mode name error")

    aug_img = np.squeeze(aug_img)
    aug_img = Image.fromarray(aug_img)

    aug_tar_img = np.squeeze(aug_tar_img)
    aug_tar_img = Image.fromarray(aug_tar_img)

    return aug_img, aug_tar_img

def get_meta_split_data_loaders(test_vendor='D'):
    random.seed(14)

    domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset,\
    domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset,\
    test_dataset = \
        get_data_loader_folder()

    return  domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset,\
            domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset, \
            test_dataset 

def get_data_loader_folder():


    print("loading labeled dateset")

    domain_1_labeled_dataset = ImageFolder(if_aug = 0, split='labeled', n_divide='1_2',  label=True, train=True)
    domain_2_labeled_dataset = ImageFolder(if_aug = 1, split='labeled', n_divide='1_2',  label=True, train=True)
    domain_3_labeled_dataset = ImageFolder(if_aug = 2, split='labeled', n_divide='1_2',  label=True, train=True)

    print("loading unlabeled dateset")
    domain_1_unlabeled_dataset = ImageFolder(if_aug = 0, split='unlabeled_label', n_divide='1_2', label=False, train=True)
    domain_2_unlabeled_dataset = ImageFolder(if_aug = 1, split='unlabeled_label', n_divide='1_2', label=False, train=True)
    domain_3_unlabeled_dataset = ImageFolder(if_aug = 2, split='unlabeled_label', n_divide='1_2', label=False, train=True)

    print("loading test dateset")
    test_dataset = ImageFolder(if_aug = 0, split='val', n_divide='1_2', label=True, train=False)

    return domain_1_labeled_dataset, domain_2_labeled_dataset, domain_3_labeled_dataset,\
           domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, domain_3_unlabeled_dataset,\
           test_dataset

def set_files(split, n_divide):
    if split == "val":
        file_list = os.path.join(prefix, split + ".txt")
        file_list = [line.rstrip().split('\t') for line in tuple(open(file_list, "r"))]
        images, labels = list(zip(*file_list))
    elif split in ["labeled", "unlabeled_label"]:
        file_list = os.path.join(prefix, n_divide, f"{split}" + ".txt")
        file_list = [line.rstrip().split('\t') for line in tuple(open(file_list, "r"))]
        images, labels = list(zip(*file_list))
    else:
        raise ValueError(f"Invalid split name {split}")
    return images, labels

class ImageFolder(data.Dataset):
    def __init__(self, if_aug = 0, split = 'val', n_divide = None, data_dir= None, mask_dir= None,  train=True, label=True, loader=default_loader):

        print("data_dirs", data_dir)
        # print("mask_dirs", mask_dir)
        self.data_dir = data_dir
        self.mask_dir = mask_dir
        self.loader = loader
        self.train = train
        self.label = label
        self.newsize = 288
        self.if_aug = if_aug

        if self.train and self.label:
            ratio = 0.2       #  20%
        else:
            ratio = 1


        self.Fourier_aug = default_config['Fourier_aug']
        self.fourier_mode = default_config['fourier_mode']
        self.alpha = 0.3

        self.images, self.labels = set_files(split, n_divide)
        print("length of images", len(self.images))

        fourier_imgs = []
        if self.train == True:
            for i in range(len(self.images)):
                fourier_path_img = os.path.join(root, self.images[i])
                fourier_imgs.append(fourier_path_img)
        self.fourier = fourier_imgs


    def __getitem__(self, index):
        # print("index=",index)
        path_img = os.path.join(root, self.images[index])
        # img = Image.open(path_img).convert('L')   # L : 灰色8bit  F : 灰色32bit浮点（傅里叶增强用
        img = Image.open(path_img).convert('F')
        h, w = img.size
        # print(h, w)


        path_mask = os.path.join(root, self.labels[index])

        # mask = Image.open(path_mask).convert('L')
        mask = Image.open(path_mask).convert('F')

        if self.if_aug==1:
            # 对比度
            # img = ImageEnhance.Contrast(img).enhance(3)
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(method=Image.FLIP_LEFT_RIGHT)
            # 边缘检测
            # img = np.asarray(img)
            # Horizontal = cv2.Sobel(img, 0, 1, 0, cv2.CV_64F)
            # Vertical = cv2.Sobel(img, 0, 0, 1, cv2.CV_64F)
            # img = cv2.bitwise_or(Horizontal, Vertical)
            # img = Image.fromarray(img)
        elif self.if_aug==2:
            img = img.transpose(method=Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(method=Image.FLIP_TOP_BOTTOM)



        # for Fourier dirs
        if self.train == True :
            fourier_paths = random.sample(self.fourier, 1)
            # fourier_img = self.loader(fourier_paths[0])
            # fourier_img = Image.fromarray(fourier_img)
            # fourier_img = Image.open(fourier_paths[0]).convert('L')
            fourier_img = Image.open(fourier_paths[0]).convert('F')
            # if self.if_aug == 1:
            #     fourier_img = fourier_img.transpose(method=Image.FLIP_LEFT_RIGHT)
            # elif self.if_aug == 2:
            #     fourier_img = fourier_img.transpose(method=Image.FLIP_TOP_BOTTOM)


        # label
        if self.label:
            # train labeled data
            if self.train:
                # rotate, random angle between 0 - 90
                # angle = random.randint(0, 90)
                # img = F.rotate(img, angle, InterpolationMode.BILINEAR)
                # mask = F.rotate(mask, angle, InterpolationMode.NEAREST)
                # if h > 110 and w > 110:
                if h > 310 and w > 310:
                    # size = (100, 100)
                    size = (300, 300)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = [transforms.Resize((self.newsize, self.newsize))] + transform_list
                    transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                    transform = transforms.Compose(transform_list)
                else:
                    size = (300, 300)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                    transform = transforms.Compose(transform_list)

                img = transform(img)
                mask = transform(mask)

                # fourier_img = F.rotate(fourier_img, angle, InterpolationMode.BILINEAR)
                fourier_img = transform(fourier_img)
                # aug_img, _ = fourier_augmentation(img, fourier_img, self.fourier_mode, self.alpha)

                # aug_img = gt_cut(img, fourier_img, mask)
                aug_img = img


                img = F.to_tensor(np.array(img))
                aug_img = F.to_tensor(np.array(aug_img))
                mask = F.to_tensor(np.array(mask))
                mask = (mask > 0.1).float()


                mask_bg = (mask.sum(0) == 0).type_as(mask)  # H,W
                mask_bg = mask_bg.reshape((1, mask_bg.size(0), mask_bg.size(1)))
                mask = torch.cat((mask, mask_bg), dim=0)
            # test data
            else:
                if h > 310 and w > 310:
                    size = (300, 300)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = [transforms.Resize((self.newsize, self.newsize))] + transform_list
                    transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                    transform = transforms.Compose(transform_list)
                else:
                    size = (100, 100)
                    transform_list = [transforms.CenterCrop(size)]
                    transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                    transform = transforms.Compose(transform_list)
                img = transform(img)
                mask = transform(mask)

                img = F.to_tensor(np.array(img))
                mask = F.to_tensor(np.array(mask))
                mask = (mask > 0.1).float()
                aug_img = torch.tensor([0])


        # train unlabel data
        else:
            # rotate, random angle between 0 - 90
            # angle = random.randint(0, 90)
            # img = F.rotate(img, angle, InterpolationMode.BILINEAR)

            if h > 310 and w > 310:
                size = (300, 300)
                transform_list = [transforms.CenterCrop(size)]
                transform_list = [transforms.Resize((self.newsize, self.newsize))] + transform_list
                transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                transform = transforms.Compose(transform_list)
            else:
                size = (300, 300)
                transform_list = [transforms.CenterCrop(size)]
                transform_list = transform_list + [transforms.Resize((self.newsize, self.newsize))]
                transform = transforms.Compose(transform_list)

            img = transform(img)

            # fourier_img = F.rotate(fourier_img, angle, InterpolationMode.BILINEAR)
            fourier_img = transform(fourier_img)
            aug_img, _ = fourier_augmentation(img, fourier_img, self.fourier_mode, self.alpha)

            # random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
            # aug_img = ImageEnhance.Contrast(img).enhance(3)  # 调整图像对比度
            # aug_img = img



            img = F.to_tensor(np.array(img))
            aug_img = F.to_tensor(np.array(aug_img))
            mask = torch.tensor([0])


        ouput_dict = dict(
            img = img,
            aug_img = aug_img,
            mask = mask,
            path_img = path_img
        )
        return ouput_dict # pytorch: N,C,H,W

    def __len__(self):
        # return len(self.imgs)
        return len(self.images)




if __name__ == '__main__':
    test_vendor = 'D'

    domain_1_labeled_dataset, domain_2_labeled_dataset, \
    domain_1_unlabeled_dataset, domain_2_unlabeled_dataset, \
    test_dataset  = get_meta_split_data_loaders(test_vendor=test_vendor)

    label_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=True, drop_last=True, pin_memory=True)

    dataiter = iter(label_loader)
    output = dataiter.next()
    img = output['img']
    mask = output['mask']
    aug_img = output['aug_img']

    print(img.shape)
    print(mask.shape)
    print(aug_img.shape)

    # torch.set_printoptions(threshold=np.inf)
    # with open('./mask.txt', 'wt') as f:
    #     print(mask, file=f)
    mask = mask[:,0,:,:]
    img = im_convert(img, False)
    # aug_img = im_convert(aug_img, False)
    mask = im_convert(mask, False)
    save_image(img, './fpic/label_'+str(default_config['fourier_mode'])+'_img.png')
    # save_image(aug_img, './fpic/label_'+str(default_config['fourier_mode'])+'_aug_img.png')
    save_image(mask, './fpic/label_'+str(default_config['fourier_mode'])+'_mask.png')
