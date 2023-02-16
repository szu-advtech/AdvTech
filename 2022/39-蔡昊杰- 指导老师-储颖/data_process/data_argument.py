from unet.utils import *
from unet.data import *
import os
import cv2
from osgeo import gdal
import numpy as np


def dataArgument(im_data, path):
    train_image_path = "/Data/train/image"
    train_label_path = "/Data/train/label"

    #  进行几何变换数据增强
    imageList = os.listdir(train_image_path)
    labelList = os.listdir(train_label_path)
    tran_num = len(imageList) + 1
    for i in range(len(imageList)):
        #  图像
        img_file = train_image_path + "\\" + imageList[i]
        im_height, im_width, im_bands, im_data, im_geotrans, im_proj = readTif(
            img_file, mode='Initialization')
        #  标签
        label_file = train_label_path + "\\" + labelList[i]
        label = cv2.imread(label_file)

        #  图像水平翻转
        im_data_hor = np.flip(im_data, axis=2)
        hor_path = train_image_path + "\\" + str(tran_num) + imageList[i][-4:]
        writeTiff(im_data_hor, im_geotrans, im_proj, hor_path)
        #  标签水平翻转
        Hor = cv2.flip(label, 1)
        hor_path = train_label_path + "\\" + str(tran_num) + labelList[i][-4:]
        cv2.imwrite(hor_path, Hor)
        tran_num += 1

        #  图像垂直翻转
        im_data_vec = np.flip(im_data, axis=1)
        vec_path = train_image_path + "\\" + str(tran_num) + imageList[i][-4:]
        writeTiff(im_data_vec, im_geotrans, im_proj, vec_path)
        #  标签垂直翻转
        Vec = cv2.flip(label, 0)
        vec_path = train_label_path + "\\" + str(tran_num) + labelList[i][-4:]
        cv2.imwrite(vec_path, Vec)
        tran_num += 1

        #  图像对角镜像
        im_data_dia = np.flip(im_data_vec, axis=2)
        dia_path = train_image_path + "\\" + str(tran_num) + imageList[i][-4:]
        writeTiff(im_data_dia, im_geotrans, im_proj, dia_path)
        #  标签对角镜像
        Dia = cv2.flip(label, -1)
        dia_path = train_label_path + "\\" + str(tran_num) + labelList[i][-4:]
        cv2.imwrite(dia_path, Dia)
        tran_num += 1
