import random

import jittor as jt
import cv2 as cv
import numpy as np
from PIL import Image
from jittor.dataset import Dataset
import jittor.transform as transforms


class MyDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, resize=224):
        super().__init__()
        self.images_path = images_path
        self.labels = images_class
        p1 = random.random()
        p2 = random.random()
        self.transform = transforms.Compose([transforms.Resize(size=(resize, resize)),
                                             transforms.RandomChoice([transforms.RandomHorizontalFlip(p1),
                                                                      transforms.RandomVerticalFlip(p2),
                                                                      transforms.RandomRotation(10, resample=False,
                                                                                                expand=False,
                                                                                                center=None),
                                                                      transforms.ColorJitter(brightness=0.5,
                                                                                             contrast=0.5, hue=0.5),
                                                                      transforms.RandomCrop(resize), ])])

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')
        img = self.transform(img)
        label = self.labels[item]
        # img = jt.var(img)
        # label = jt.var(label)
        # if self.transform is not None:
        #     img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.images_path)

    @staticmethod
    def collate_batch(batch):
        images, labels = tuple(zip(*batch))
        images = jt.stack(images, dim=0)
        labels = jt.array(labels)
        return images, labels
