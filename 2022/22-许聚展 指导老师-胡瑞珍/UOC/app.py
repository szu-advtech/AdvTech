#!/usr/bin/env python3

# Copyright (c) 2020 NVIDIA Corporation. All rights reserved.
# This work is licensed under the NVIDIA Source Code License - Non-commercial. Full
# text can be found in LICENSE.md

"""Test a PoseCNN on images"""

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data

import os, sys
import numpy as np
from PIL import Image
import json
from scipy.spatial.transform import Rotation

import UOC.tools._init_paths
from fcn.test_module import img_segment, img_process, compute_xyz, depth_to_pc
import networks as networks 

UOC_path = "E:/workspace/visual_match/UOC"

def get_cam():
    filename = f'{UOC_path}/data/demo/camera_params.json'
    if os.path.exists(filename):
        with open(filename) as f:
            camera_params = json.load(f)
    else:
        camera_params = None
    return camera_params



def get_network():
    
    gpu_id = 0
    pretrained =  f'{UOC_path}/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_sampling_epoch_16.checkpoint.pth'
    pretrained_crop = f'{UOC_path}/data/checkpoints/seg_resnet34_8s_embedding_cosine_rgbd_add_crop_sampling_epoch_16.checkpoint.pth'
    # pretrained =  f'{UOC_path}/data/checkpoints/seg_resnet34_8s_embedding_cosine_color_sampling_epoch_16.checkpoint.pth'
    # pretrained_crop = f'{UOC_path}/data/checkpoints/seg_resnet34_8s_embedding_cosine_color_crop_sampling_epoch_16.checkpoint.pth'

    network_name = 'seg_resnet34_8s_embedding'

    # device
    device = torch.device('cuda:{:d}'.format(gpu_id))
    num_classes = 2
    train_num_unit = 64
    
    network_data = torch.load(pretrained)

    network = networks.__dict__[network_name](num_classes, train_num_unit, network_data).cuda(device=device)
    network = torch.nn.DataParallel(network, device_ids=[gpu_id]).cuda(device=device)
    cudnn.benchmark = True
    network.eval()

    network_data_crop = torch.load(pretrained_crop)
    network_crop = networks.__dict__[network_name](num_classes, train_num_unit, network_data_crop).cuda(device=device)
    network_crop = torch.nn.DataParallel(network_crop, device_ids=[gpu_id]).cuda(device=device)
    network_crop.eval()

    return network, network_crop, device

class Segmenter(object):
    def __init__(self) -> None:
        
        network, network_crop, device = get_network()

        self.network = network
        self.network_crop = network_crop        
        self.device = device
    
    def segment(self, rgb: np.array, dep:np.array, camera_params:dict):
        # dep is meter
        rgb_batch, dep_batch = img_process(rgb, dep, camera_params)

        out_label, out_label_refined = img_segment(rgb_batch, dep_batch, self.network, self.network_crop, self.device, False)
        
        return out_label[0], out_label_refined[0]
    
    def crop(self, rgb, label):
        all_ids = np.unique(label)

        all_imgs = []
        bboxes = []
        for i in list(all_ids):
            if i == 0: continue

            if torch.sum(label == i) < 32*32:
                continue

            x, y = np.where(label == i)

            min_x = x.min()
            max_x = x.max()
            min_y = y.min()
            max_y = y.max()

            if (max_x - min_x) * (max_x - min_x) > 250**2:
                continue

            img = rgb[min_x:max_x, min_y:max_y]
            all_imgs.append(img)
            bboxes.append([min_x,max_x, min_y,max_y])
        return all_imgs, bboxes

    def segment_and_crop(self, rgb: np.array, dep:np.array, camera_params:dict):
        label, label_refined = self.segment(rgb, dep, camera_params)
        all_rgb, bbox = self.crop(rgb, label)
        return all_rgb, bbox

    def crop_point_cloud(self, dep, camera_params, bboxes):

        c2w = camera_params['c2w']
        cam = camera_params['cam']
        
        # 因为 omniverse 里面的相机坐标系，和相机在世界坐标系中的坐标系是不一样的，相机朝向是Z轴负方向
        # 所以 深度图计算出在相机坐标系内的点后，需要两步的坐标系转换，
        # 从相机坐标系 -->  在世界坐标系中的相机坐标系（也就是绕x轴转180） --> 世界坐标系

        rot_x = Rotation.from_rotvec(np.pi * np.array([1,0,0])).as_matrix()
        c2w[:3,:3] = c2w[:3,:3] @ rot_x
        
        pc = depth_to_pc(dep, cam, c2w)
        
        ret = []

        for bbox in bboxes:
            min_x, max_x, min_y, max_y = bbox
            points = pc[min_x:max_x, min_y:max_y]
            points = points.reshape(-1,3)
            ret.append(points)

        return ret

if __name__ == '__main__':
    
    np.random.seed(3)

    # list images
    images_color = []
    images_depth = []
    
    network, network_crop, device = get_network()
    
    
    # camera_params = get_cam()
    # rgb = cv2.imread("./data/demo/000002-color.png", cv2.COLOR_BGR2RGB)
    # dep = cv2.imread("./data/demo/000002-depth.png", cv2.IMREAD_ANYDEPTH)
    # dep = dep.astype(np.float32) / 1000.0

    # rgb = cv2.imread("../robot/images/sc_rgb.png", cv2.COLOR_BGR2RGB)
    rgb = np.array(Image.open("../robot/images/sc_rgb.png"))[:,:,:3][:,:,::-1].copy()
    dep = np.load("../robot/images/sc_dep.npy")
    cam = np.load("../robot/images/sc_cam.npy")
    camera_params = {}
    camera_params['x_offset'] = cam[0,0]
    camera_params['y_offset'] = cam[1,1]
    camera_params['fx'] = cam[0,2]
    camera_params['fy'] = cam[1,2]

    # dep = None
    rgb_batch, dep_batch = img_process(rgb, dep, camera_params)

    rgb_batch = torch.cat([rgb_batch], dim=0)
    dep_batch = torch.cat([dep_batch], dim=0)
    # if dep_batch is not None:
    
    out_label, out_label_refined = img_segment(rgb_batch, dep_batch, network, network_crop, device, True)

