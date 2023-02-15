import torch
import torch.nn as nn
import torchvision
from . import resnet, resnext, mobilenet, hrnet
from lib.nn import SynchronizedBatchNorm2d
from torch.autograd import Variable
BatchNorm2d = SynchronizedBatchNorm2d
import numpy as np

if __name__ == '__main__':
    pass