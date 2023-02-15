import os
import glob
import matplotlib.pyplot as plt

import scipy.misc
import scipy.ndimage
import numpy as np
import cv2 as cv
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# 改变图像尺寸函数
def inverse_transform(images):
  return (images+1.)/2
# 根据文件夹读取里面的所有图像函数
def prepare_data(sess, dataset):
  filenames = os.listdir(dataset)
  data_dir = os.path.join(os.getcwd(), dataset)
  data = glob.glob(os.path.join(data_dir, "*.png"))
  data = data + glob.glob(os.path.join(data_dir, "*.jpg"))
  return data

def imread(path, is_grayscale=False):
  if is_grayscale:
    return scipy.misc.imread(path, flatten=True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def imsave(image, path):
  imsaved = (inverse_transform(image)).astype(np.float)
  return scipy.misc.imsave(path, imsaved)

def get_image(image_path,is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return image/255
def get_lable(image_path,is_grayscale=False):
  image = imread(image_path, is_grayscale)
  return image/255.

# 保存图像函数
def imsave_lable(image, path):
  # print(path)
  return scipy.misc.imsave(path, image*255)

def white_balance(img, percent=0):
    # img=cv.cvtColor(img, cv.COLOR_RGB2BGR)
    out_channels = []
    cumstops = (
        img.shape[0] * img.shape[1] * percent / 200.0,
        img.shape[0] * img.shape[1] * (1 - percent / 200.0)
    )
    for channel in cv.split(img):
        cumhist = np.cumsum(cv.calcHist([channel], [0], None, [256], (0,256)))
        low_cut, high_cut = np.searchsorted(cumhist, cumstops)
        lut = np.concatenate((
            np.zeros(low_cut),
            np.around(np.linspace(0, 255, high_cut - low_cut + 1)),
            255 * np.ones(255 - high_cut)
        ))
        out_channels.append(cv.LUT(channel, lut.astype('uint8')))
    img=cv.merge(out_channels)
    print(percent)
    return img

def adjust_gamma(image, gamma=0.7):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv.LUT(image.astype(np.uint8), table.astype(np.uint8))