from skimage import io, morphology
import skimage
import cv2
import numpy as np


def gen_edge_mask(mask, kernel_size=5):
    kernel = skimage.morphology.disk(kernel_size)
    # morphology 形态学操作，如开闭运算、骨架提取等
    # erosion 返回图像的灰度形态侵蚀。形态侵蚀将 (i,j) 处的像素设置为以 (i,j) 为中心的邻域中所有像素的最小值。侵蚀缩小了明亮区域并扩大了黑暗区域。
    # print("mask", mask.shape)
    # print("mask", mask.shape)
    erosion_mask = morphology.erosion(mask, kernel)
    edge_mask = erosion_mask - mask

    return edge_mask // 255


if __name__ == '__main__':
    img_path = '/home/liaogy/lgy/mvssnet2/samples/86.jpg'
    mask_path = '/home/liaogy/lgy/mvssnet2/samples/86.png'

    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2GRAY) // 255
    edge_mask = gen_edge_mask(mask, kernel_size=2)
    print(np.unique(edge_mask))
    # cv2.imwrite('./samples/86_edge.png', edge_mask * 255)