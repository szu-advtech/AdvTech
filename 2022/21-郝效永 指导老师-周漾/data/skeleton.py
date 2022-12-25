import os
import cv2
import torch
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms

def __crop(img,pos):
    y1= pos
    return img.crop((0,y1-64,128,y1 + 64))
    pass

def __cropcenter(img,target_h):
    return img.crop((int((target_h)/2)-64, 0, int(target_h/2)+64, target_h))
    pass

def Totensor_Image_label(img,crop_center,target_h):
        
        transforms_label = transforms.Compose([
            transforms.Resize(target_h,interpolation=Image.BILINEAR),
            transforms.Lambda(lambda img: __cropcenter(img,target_h)),
            transforms.Lambda(lambda img: __crop(img,crop_center)),
            transforms.ToTensor(),
            transforms.Normalize((0.5),(0.5))
        ])
        one_pice = transforms_label(img)
        return one_pice
        pass

def get_label_tensor(path_candidate,path_subset,crop_center = None,target_h = None,is_seg = False):
        candidate = np.loadtxt(path_candidate)
        subset = np.loadtxt(path_subset)
        stickwidth = 20
        limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
                [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
                [1, 16], [16, 18], [3, 17], [6, 18]]
        colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
                [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
                [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]
        canvas = np.zeros((1024, 1024, 3), dtype=np.uint8)
        cycle_radius = 20
        for i in range(18):
            index = int(subset[i])
            if index == -1:
                continue
            x, y = candidate[index][0:2]
            cv2.circle(canvas, (int(x), int(y)), cycle_radius, colors[i], thickness=-1)
        joints = []
        for i in range(17):
            index = subset[np.array(limbSeq[i]) - 1]
            cur_canvas = canvas.copy()
            if -1 in index:
                joints.append(np.zeros_like(cur_canvas[:, :, 0]))
                continue
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly((int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
            cv2.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)
            joint = np.zeros_like(cur_canvas[:, :, 0])
            cv2.fillConvexPoly(joint, polygon, 255)
            joint = cv2.addWeighted(joint, 0.4, joint, 0.6, 0)
            joints.append(joint)
        #pose = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        pose = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)).resize((1024, 1024), resample=Image.NEAREST)
        #params = get_params(self.opt, pose.size)
        #transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        #transform_img = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
        tensors_dist = 0
        e = 1
        for i in range(len(joints)):
            im_dist = cv2.distanceTransform(255-joints[i], cv2.DIST_L1, 3)
            im_dist = np.clip((im_dist/3), 0, 255).astype(np.uint8)
            tensor_dist = Totensor_Image_label(Image.fromarray(im_dist),crop_center=crop_center,target_h=target_h)
            
            tensors_dist = tensor_dist if e == 1 else torch.cat([tensors_dist, tensor_dist])
            e += 1
        tensor_pose = Totensor_Image_label(pose,crop_center=crop_center,target_h=target_h)
        label_tensor = torch.cat((tensor_pose, tensors_dist), dim=0)
        return pose,label_tensor
        