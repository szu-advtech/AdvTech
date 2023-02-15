# -*- coding: utf-8 -*-
"""
License: MIT
@author: gaj
E-mail: anjing_guo@hnu.edu.cn
"""

import numpy as np
import cv2
import os
import spectral as spy
import scipy.io as sio

from tqdm import tqdm

from dataloader import DataSet

from MTF_GLP import MTF_GLP
from MTF_GLP_HPM import MTF_GLP_HPM
from GSA import GSA
from CNMF import CNMF
from PNN import PNN

from metrics import ref_evaluate

'''load data'''
used_ms = np.squeeze(sio.loadmat(r'..\data\pavia\LRHS.mat')['b'].astype(np.float32)).transpose((1, 2, 0))
used_pan = sio.loadmat(r'..\data\pavia\PAN.mat')['b'].astype(np.float32).transpose((1, 2, 0))
gt = np.squeeze(sio.loadmat(r'..\data\pavia\gtHS.mat')['b'].astype(np.float32)).transpose((1, 2, 0))


'''evaluating all methods'''
ref_results = {}
ref_results.update({'metrics: ': '    CC,     SAM,    RMSE,    ERGAS'})

'''setting save parameters'''
rgb = {"pavia": [55, 30, 5]}
set_name = "pavia"
save_images = True  # 是否保存结果图片
save_channels = [0, 1, 2]  # BGR-NIR for GF2
save_dir = '../results/'+set_name
if save_images and (not os.path.isdir(save_dir)):
    os.makedirs(save_dir)

'''MTF_GLP method'''
print("evaluating MTF_GLP method...")
fused_image = MTF_GLP(used_pan[:, :, :], used_ms[:, :, :])
ref = ref_evaluate(fused_image, gt, used_ms)
ref_results.update({'MTF_GLP    ': ref})
# save
if save_images:
    spy.save_rgb(save_dir + '/MTF_GLP.jpg', fused_image, rgb[set_name])

'''MTF_GLP_HPM method'''
print("evaluating MTF_GLP_HPM method...")
fused_image = MTF_GLP_HPM(used_pan[:, :, :], used_ms[:, :, :])
ref = ref_evaluate(fused_image, gt, used_ms)
ref_results.update({'MTF_GLP_HPM': ref})
# save
if save_images:
    spy.save_rgb(save_dir + '/MTF_GLP_HPM.jpg', fused_image, rgb[set_name])


'''GSA method'''
print("evaluating GSA method...")
fused_image = GSA(used_pan[:, :, :], used_ms[:, :, :])
ref = ref_evaluate(fused_image, gt, used_ms)
ref_results.update({'GSA        ': ref})
# save
if save_images:
    spy.save_rgb(save_dir + '/GSA.jpg', fused_image, rgb[set_name])

'''CNMF method'''
print("evaluating CNMF method...")
fused_image = CNMF(used_pan[:, :, :], used_ms[:, :, :])
ref = ref_evaluate(fused_image, gt, used_ms)
ref_results.update({'CNMF       ': ref})
# save
if save_images:
    spy.save_rgb(save_dir + '/CNMF.jpg', fused_image, rgb[set_name])


'''PNN method'''
# print('evaluating PNN method')
# fused_image = PNN(used_pan[:, :, :], used_ms[:, :, :])
# ref = ref_evaluate(fused_image, gt, used_ms)
# ref_results.update({'PNN        ': ref})
# # save
# if save_images:
#     spy.save_rgb(save_dir + '/PNN.jpg', fused_image, rgb[set_name])


''''print result'''
print('################## reference comparision #######################')
for index, i in enumerate(ref_results):
    if index == 0:
        print(i, ref_results[i])
    else:
        print(i, [round(j, 4) for j in ref_results[i]])
print('################## reference comparision #######################')
