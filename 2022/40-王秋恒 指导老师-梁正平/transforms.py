import math
import random
from typing import Tuple

import numpy as np
import torch
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt

class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target



class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target):
        image = F.to_tensor(image)
        target = F.to_tensor(target).permute(1, 2, 0)
        return image, target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)

        #target = F.normalize(target, mean=self.mean,std=self.std)
        return image, target