import argparse
import os
from math import log10
import scipy.io as scio
import numpy as np
import pandas as pd
import torch
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_utils import TrainDatasetFromFolder
from model import Discriminator
from utils import AverageMeter, SROCC, PLCC, RMSE
from utils import SimpleProgressBar as ProgressBar

if __name__ == '__main__':
    MODEL_NAME = 'netD_epoch_50.pth'
    CROP_SIZE=128
    times=200
    t_patch_path = '../datasets/SIQAD_4Patches/DistortedImages'
    v_patch_path = '../datasets/SIQAD_4Patches/DistortedImages'
    t_data_path = './imagedata/DMOS_SIQAD.mat'
    v_data_path = './imagedata/DMOS_SIQAD.mat'
    t_si, t_sj = 49,7
    v_si, v_sj = 49,7
    train_index = (1,2,3,4,5,6,7,8,9,12,13,14,16,17,19,20)
    val_index = (10,11,15,18)
    test_index = val_index

    model = Discriminator().eval()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load('epochs/' + MODEL_NAME))
    test_set = TrainDatasetFromFolder(v_patch_path, crop_size=CROP_SIZE,patches=1,istrain=False, data_path= v_data_path, si=v_si, sj=v_sj,indexlist=test_index)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    srocc = SROCC()
    plcc = PLCC()
    rmse = RMSE()
    len_test = len(test_loader)
    pb = ProgressBar(len_test, show_step=True)
    test_bar = tqdm(test_loader, desc='[testing benchmark datasets]')
    i = 0
    for test_data1,test_data2,test_data3,test_data4,label,type in test_bar:
        i+=1
        with torch.no_grad():
            test_z1 = Variable(test_data1)
            test_z2 = Variable(test_data2)
            test_z3 = Variable(test_data3)
            test_z4 = Variable(test_data4)
            if torch.cuda.is_available():
                test_z1 = test_z1.cuda()
                test_z2 = test_z2.cuda()
                test_z3 = test_z3.cuda()
                test_z4 = test_z4.cuda()
            

            _,pred,_ = model(test_z1, test_z2, test_z3, test_z4,test_z1, test_z2, test_z3, test_z4)
            output = pred.cpu().squeeze().data.numpy()
            score = label.data.numpy()
            srocc.update(score, output)
            plcc.update(score, output)
            rmse.update(score, output)
            
            pb.show(i, "Test: [{0:5d}/{1:5d}]\t"
                    "Score: {2:.4f}\t"
                    "Label: {3:.4f}"
                    .format(i+1, len_test, float(output), float(score)))
            
    print("\n\nSROCC: {0:.4f}\n"
            "PLCC: {1:.4f}\n"
            "RMSE: {2:.4f}"
            .format(srocc.compute(), plcc.compute(), rmse.compute())
    )
