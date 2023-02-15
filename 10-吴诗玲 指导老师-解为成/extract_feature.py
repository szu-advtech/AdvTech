import numpy as np
import os
import sys
from ctypes import *
import time
import pandas as pd
from numpy.ctypeslib import ndpointer
import numpy.ctypeslib as npct
from scipy import signal
import math
import combine_files
import select_feature

import time

def find_feature_name_2019():
    feature_name=[' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z', ' pose_Tx', ' pose_Ty', ' pose_Tz', ' pose_Rx', ' pose_Ry', ' pose_Rz', ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']
    feature_name = [x.strip() for x in feature_name]
    return feature_name

def find_feature_name():
    feature_name=[' gaze_0_x', ' gaze_0_y', ' gaze_0_z', ' gaze_1_x', ' gaze_1_y', ' gaze_1_z', ' pose_Tx', ' pose_Ty', ' pose_Tz', ' pose_Rx', ' pose_Ry', ' pose_Rz', ' AU01_r', ' AU02_r', ' AU04_r', ' AU05_r', ' AU06_r', ' AU07_r', ' AU09_r', ' AU10_r', ' AU12_r', ' AU14_r', ' AU15_r', ' AU17_r', ' AU20_r', ' AU23_r', ' AU25_r', ' AU26_r', ' AU45_r']
    return feature_name


def preprocess(x):
    feature_num,frames= x.shape
    return x-np.median(x,axis=1).reshape(feature_num,-1)

def getVideoFeature(feaVec):
    feaNum = feaVec.shape[0]
    videoFea = np.zeros([feaNum * 12], dtype=np.float32)

    mean_v = np.mean(feaVec,axis = 1)
    std_v = np.std(feaVec, axis=1,ddof = 1)
    max_v = np.max(feaVec, axis=1)
    min_v = np.min(feaVec, axis=1)

    diff_v = np.diff(feaVec,axis = 1 ) #一阶差分

    mean_diff_v = np.mean(diff_v,axis = 1)
    std_diff_v =  np.std(diff_v, axis=1,ddof = 1)
    max_diff_v =  np.max(diff_v, axis=1)
    min_diff_v =  np.min(diff_v, axis=1)

    a_b = feaVec[:,0:feaVec.shape[1]-2] - feaVec[:,2:]  #二阶差分
    a_b_mean = np.mean(a_b,axis = 1)
    a_b_std = np.std(a_b, ddof=1,axis = 1)
    a_b_max = np.max(a_b, axis=1)
    a_b_min = np.min(a_b, axis=1)


    for i in range(feaNum):
        fea_patch_idx = i * 12
        videoFea[fea_patch_idx ] = mean_v[i]
        videoFea[fea_patch_idx + 1] = std_v[i]
        videoFea[fea_patch_idx + 2] = max_v[i]
        videoFea[fea_patch_idx + 3] = min_v[i]

        videoFea[fea_patch_idx + 4] = mean_diff_v[i]
        videoFea[fea_patch_idx + 5] = std_diff_v[i]
        videoFea[fea_patch_idx + 6] = max_diff_v[i]
        videoFea[fea_patch_idx + 7] = min_diff_v[i]

        videoFea[fea_patch_idx + 8] = a_b_mean[i]
        videoFea[fea_patch_idx + 9] = a_b_std[i]
        videoFea[fea_patch_idx + 10] = a_b_max[i]
        videoFea[fea_patch_idx + 11] = a_b_min[i]
    return videoFea

def cut_videos(x,fre):
    len=x.shape[1]
    if len<fre:
        x=np.pad(x, ((0, 0), (0, fre)), 'constant')
        len=fre
    raw_num_keep_frame=math.floor(len/fre)
    num_keep_frame=raw_num_keep_frame*fre
    num_delete=len-num_keep_frame

    num_delete_start=math.ceil(num_delete/2)
    num_delete_end=math.floor(num_delete/2)

    x=x[:,num_delete_start:len-num_delete_end]

    return x,num_keep_frame,raw_num_keep_frame


def fourier_transform_select(x,N,num_multiple,fre):
    channel_num,length=x.shape
    common_temp_amp=np.zeros((channel_num,fre))
    common_temp_pha=np.zeros((channel_num,fre))

    all_data = np.fft.fft(x,axis = 1)
    amp_map= np.abs(all_data)/length
    phase_map = np.angle(all_data)

    for i in range(fre):
        common_temp_amp[:,i]=amp_map[:,i*num_multiple]
        common_temp_pha[:,i]=phase_map[:,i*num_multiple]

    amp_m=amp_map[:,math.floor(length/2)].reshape(channel_num,1)
    phase_m=phase_map[:,math.floor(length/2)].reshape(channel_num,1)
    amp_map_return=np.concatenate((amp_m,common_temp_amp[:,:N-1]),axis=1)
    phase_map_return=np.concatenate((phase_m,common_temp_pha[:,:N-1]),axis=1)

    return amp_map_return,phase_map_return



def rebuildFeature_select(input,N_,fre,t_length_ ):
    data,num_keep_frame,num_multiple=cut_videos(input,fre)
    processed_data=preprocess(data)
    sta_fea=getVideoFeature(processed_data)
    amp_map,phase_map=fourier_transform_select(processed_data,N_,num_multiple,fre)
    amp_map_flat = amp_map.reshape(t_length_)
    phase_map_flat = phase_map.reshape(t_length_)

    return np.concatenate((sta_fea, amp_map_flat, phase_map_flat), axis=0)

def zeropadding(x):
    long=7400
    x=np.pad(x, ((0, 0), (0, long)), 'constant')
    return x

def fourier_transform_zero(x,N):
    channel_num,length=x.shape

    all_data = np.fft.fft(x,axis = 1) #进行傅里叶变换
    amp_map= np.abs(all_data)/length
    phase_map = np.angle(all_data)


    amp_map_return=amp_map[:,:N]
    phase_map_return=phase_map[:,:N]

    return amp_map_return,phase_map_return


def rebuildFeature_zero(input,N_,t_length_):
    data=zeropadding(input)
    processed_data=preprocess(data)
    sta_fea=getVideoFeature(processed_data)
    amp_map,phase_map=fourier_transform_zero(processed_data,N_)
    amp_map_flat = amp_map.reshape(t_length_)
    phase_map_flat = phase_map.reshape(t_length_)

    return np.concatenate((sta_fea, amp_map_flat, phase_map_flat), axis=0)

def fourier_transform_resample(x,N,num_fre):
    all_data = np.fft.fft(x,axis = 1) #进行傅里叶变换
    temp_contain = None
    if x.shape[1]%2 == 0:
        temp_contain = all_data[:, 0: math.floor(x.shape[1] / 2) + 1] #向下取整
    else:
        temp_contain = all_data[:, 0: math.floor((x.shape[1]+1)/2)]
    temp_resample_data = signal.resample_poly(temp_contain,num_fre,temp_contain.shape[1],axis=1)
    amp_map_return =  (np.abs(temp_resample_data)/x.shape[1])[:,0:N]
    phase_map_return =  (np.angle(temp_resample_data))[:,0:N]
    return amp_map_return,phase_map_return

def rebuildFeature_resample(input,N_=64,t_length_ = 64*29):
    test_data2 = preprocess(input)
    sta_train_fea = getVideoFeature(test_data2)
    amp_map, phase_map = fourier_transform_resample(test_data2, N_, 200)
    amp_map_flat = amp_map.reshape(t_length_)
    phase_map_flat = phase_map.reshape(t_length_)
    return np.concatenate((sta_train_fea, amp_map_flat, phase_map_flat), axis=0)

def interpolation(x,method):
    x.interpolate(method=method,limit_direction='both',axis=0,inplace=True)
    return x

import os

def files_deal_AVEC2014(file_dir,save_dir,fre,N,t_length,task):
    feature_name = find_feature_name()
    listdir=os.listdir(file_dir)
    j=task
    for i in listdir:
        path_read=os.path.join(file_dir,i,j)
        files=os.listdir(path_read)
        path_save=os.path.join(save_dir,i,j)
        os.makedirs(path_save, exist_ok=True)
        for file in files:
            if os.path.splitext(file)[1] == '.csv':
                openface_file=os.path.join(path_read,file)
                data = pd.read_csv(openface_file)
                data=data[feature_name]
                data=np.array(data).T
                result=rebuildFeature_select(data,N,fre,t_length)
                save_file=os.path.join(path_save,os.path.splitext(file)[0]+'.npy')
                np.save(save_file, result)

def files_deal_AVEC2019(file_dir,save_dir,fre,N,t_length):
    feature_name = find_feature_name_2019()
    listdir=os.listdir(file_dir)
    for i in listdir:
        path_read=os.path.join(file_dir,i)
        files=os.listdir(path_read)
        path_save=os.path.join(save_dir,i)
        os.makedirs(path_save, exist_ok=True)
        for file in files:
            openface_file=os.path.join(path_read,file)
            data = pd.read_csv(openface_file)
            data=data[feature_name]
            data=np.array(data).T
            result=rebuildFeature_select(data,N,fre,t_length)
            save_file=os.path.join(path_save,os.path.splitext(file)[0]+'.npy')
            np.save(save_file, result)

if __name__ == "__main__":
    #files_deal_AVEC2014('AVEC2014/Openface_delete','AVEC2014/features_delete', 100, 50,50*29,'Freeform')
    files_deal_AVEC2019('AVEC2019/Openface_delete','AVEC2019/features_delete',500, 200, 200*29)






