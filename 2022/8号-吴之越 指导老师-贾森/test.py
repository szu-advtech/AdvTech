import torch
import torch.nn as nn
import torch.nn.functional as F
from model import *
from utils import *
import h5py
import json
import numpy as np
from scipy.io import savemat

config = json.load(open('config.json'))

resnet = ResNet(config, ResidualBlock, [2, 2, 2, 2])
linear_encoder = Linear_unmixing_encoder(config)
linear_decoder = Linear_unmixing_decoder(config)

resnet.load_state_dict(torch.load('./model_weight/resnet.pth'))
linear_encoder.load_state_dict(torch.load('./model_weight/linear_unmixing_encoder.pth'))
linear_decoder.load_state_dict(torch.load('./model_weight/linear_unmixing_decoder.pth'))

resnet.to('cpu')
linear_encoder.to('cpu')
linear_decoder.to('cpu')

img = np.transpose(np.array(h5py.File('./datasets/chikusei/chikusei/chikusei.mat')['chikusei']), (0, 2, 1))
img = torch.from_numpy(img)
img_max = torch.max(img)
img_min = torch.min(img)
img = (img - img_min) / (img_max - img_min)
img.requires_grad = False
img = img.unsqueeze(0).float()[:, :, :2500, :2300]
img_y, img_x = img, img
img_x = F.interpolate(img_x, scale_factor=0.25).float()
# img_x = img.float().unsqueeze(0)
img_x = linear_encoder(img_x)
img_x = resnet(img_x)
img_x = linear_decoder(img_x)
img_x[img_x > 1] = 1
img_x[img_x < 0] = 0
pnsr_ = pnsr(img_x, img_y)
egras_ = egras(img_x, img_y, scale_factor=4)
rmse_ = rmse(img_x, img_y)
sam_ = sam(img_x, img_y)

print(pnsr_.item(), egras_.item(), rmse_.item(), sam_.item())

img_x = img_x.detach().numpy().squeeze()
img_x = img_x.transpose(1, 2, 0)
print(img_x.shape)
savemat('./from_model_chikusei_up.mat', {'chikusei': img_x})
