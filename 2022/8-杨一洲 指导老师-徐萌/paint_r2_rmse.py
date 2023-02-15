
import re

import numpy
import numpy as np
import torch
import torch.nn as nn

import os

import torch.utils.data
from torch.utils.data import DataLoader
import torch.optim as optim
import sklearn.metrics  # 模型性能评价模块

import matplotlib.pyplot as plt

depth_list = open('depth.txt').readlines()[:20]
depth_list = list(map(lambda x: int(x), depth_list))

res = torch.load('res.ls')
x = depth_list
r2, rmse = zip(*res)
x = x[1:]
r2 = r2[1:]
rmse = rmse[1:]
x.reverse()
_r2 = list(r2)
_rmse = list(rmse)
_r2.reverse()
_rmse.reverse()
plt.figure(figsize=(4.05, 4.90))
plt.plot(_r2, x)
plt.xlabel("r2")
plt.ylabel("level")
plt.title("R2")

plt.gca().invert_yaxis()
plt.show()

plt.figure(figsize=(4, 5))
plt.plot(_rmse, x)
plt.title("NRMSE")
plt.xlabel("nrmse")
plt.ylabel("level")
plt.gca().invert_yaxis()
plt.show()