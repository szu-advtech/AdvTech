import numpy as np
import cv2 as cv
import pandas as pd
import scipy
from scipy import stats
import matplotlib.pyplot as plt
import skimage
from skimage import transform
from skimage.transform import radon
from skimage import io
from numpy.polynomial import chebyshev
import mahotas
from skimage.feature import graycomatrix, graycoprops
from skimage import io, color, img_as_ubyte
import csv
from cgitb import grey
from torch.utils.data import Dataset
import PIL
import os
import re
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import glob
from PIL import Image



#1均值
def Meanofimg (path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = img.tolist()
    mean_of_img = np.mean(img)
    return mean_of_img

#2方差
def Varianceofimg(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = img.tolist()
    variance_of_img = np.var(img)
    return variance_of_img

#3偏度
def Skew(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = img.tolist()
    skew = stats.skew(img)
    return np.sum(skew)

#4峰度
def Kurtosis(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = img.tolist()
    kurtosis = stats.kurtosis(img)
    return np.sum(kurtosis)



#5对比度
def Contrast(path):
    img = cv.imread(path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_image,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)
    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity', 'ASM']
    contrast = graycoprops(glcm, properties[0])
    return np.mean(contrast)


#6能量
def Energy(path):
    img = cv.imread(path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_image,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)
    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity', 'ASM']
    energy = graycoprops(glcm, properties[1])
    return np.mean(energy)

#7同质度
def Homogeneity(path):
    img = cv.imread(path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_image,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)
    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity', 'ASM']
    homogeneity = graycoprops(glcm, properties[2])  # 同质度
    return np.mean(homogeneity)

#8相关性
def Correlation(path):
    img = cv.imread(path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_image,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)
    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity', 'ASM']
    correlation = graycoprops(glcm, properties[3])  # 相关性
    return np.mean(correlation)

#9非相关性
def Dissimilarity(path):
    img = cv.imread(path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_image,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)
    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity', 'ASM']
    dissimilarity = graycoprops(glcm, properties[4])  # 非相似性
    return np.mean(dissimilarity)

# 10Angular Second Moment,第二角力矩
def Asm(path):
    img = cv.imread(path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_image,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)
    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity', 'ASM']
    asm = graycoprops(glcm, properties[5])  # Angular Second Moment,第二角力矩

    return np.mean(asm)

#11熵
def Entropy(path):
    img = cv.imread(path)
    gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    distances = [1, 2, 3]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    glcm = graycomatrix(gray_image,
                        distances=distances,
                        angles=angles,
                        symmetric=True,
                        normed=True)
    properties = ['contrast', 'energy', 'homogeneity', 'correlation', 'dissimilarity', 'ASM']
    pnorm = glcm / np.sum(glcm, axis=(0, 1)) + 1. / 5 ** 2
    entropy = np.sum(-pnorm * np.log(pnorm), axis=(0, 1))  # 熵
    return np.mean(entropy)

#12粗糙度
def Coarseness(path):
    image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    kmax = 5
    image = np.array(image)
    w = image.shape[0]
    h = image.shape[1]
    kmax = kmax if (np.power(2,kmax) < w) else int(np.log(w) / np.log(2))
    kmax = kmax if (np.power(2,kmax) < h) else int(np.log(h) / np.log(2))
    average_gray = np.zeros([kmax,w,h])
    horizon = np.zeros([kmax,w,h])
    vertical = np.zeros([kmax,w,h])
    Sbest = np.zeros([w,h])

    for k in range(kmax):
        window = np.power(2,k)
        for wi in range(w)[window:(w-window)]:
            for hi in range(h)[window:(h-window)]:
                average_gray[k][wi][hi] = np.sum(image[wi-window:wi+window, hi-window:hi+window])
        for wi in range(w)[window:(w-window-1)]:
            for hi in range(h)[window:(h-window-1)]:
                horizon[k][wi][hi] = average_gray[k][wi+window][hi] - average_gray[k][wi-window][hi]
                vertical[k][wi][hi] = average_gray[k][wi][hi+window] - average_gray[k][wi][hi-window]
        horizon[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))
        vertical[k] = horizon[k] * (1.0 / np.power(2, 2*(k+1)))

    for wi in range(w):
        for hi in range(h):
            h_max = np.max(horizon[:,wi,hi])
            h_max_index = np.argmax(horizon[:,wi,hi])
            v_max = np.max(vertical[:,wi,hi])
            v_max_index = np.argmax(vertical[:,wi,hi])
            index = h_max_index if (h_max > v_max) else v_max_index
            Sbest[wi][hi] = np.power(2,index)

    fcrs = np.mean(Sbest)

    return fcrs


#13对比度
def Contrast(path):
    img1 = cv.imread(path, cv.IMREAD_GRAYSCALE)
    m, n = img1.shape
    #图片矩阵向外扩展一个像素
    img1_ext = cv.copyMakeBorder(img1,1,1,1,1,cv.BORDER_REPLICATE) / 1.0   # 除以1.0的目的是uint8转为float型，便于后续计算
    rows_ext,cols_ext = img1_ext.shape
    b = 0.0
    for i in range(1,rows_ext-1):
        for j in range(1,cols_ext-1):
            b += ((img1_ext[i,j]-img1_ext[i,j+1])**2 + (img1_ext[i,j]-img1_ext[i,j-1])**2 +
                    (img1_ext[i,j]-img1_ext[i+1,j])**2 + (img1_ext[i,j]-img1_ext[i-1,j])**2)

    cg = b/(4*(m-2)*(n-2)+3*(2*(m-2)+2*(n-2))+2*4) #对应上面48的计算公式

    return cg


#14方向度
def Directionality(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    image = np.array(img, dtype = 'int64')
    h = image.shape[0]
    w = image.shape[1]
    convH = np.array([[-1,0,1],[-1,0,1],[-1,0,1]])
    convV = np.array([[1,1,1],[0,0,0],[-1,-1,-1]])
    deltaH = np.zeros([h,w])
    deltaV = np.zeros([h,w])
    theta = np.zeros([h,w])

    # calc for deltaH
    for hi in range(h)[1:h-1]:
        for wi in range(w)[1:w-1]:
            deltaH[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convH))
    for wi in range(w)[1:w-1]:
        deltaH[0][wi] = image[0][wi+1] - image[0][wi]
        deltaH[h-1][wi] = image[h-1][wi+1] - image[h-1][wi]
    for hi in range(h):
        deltaH[hi][0] = image[hi][1] - image[hi][0]
        deltaH[hi][w-1] = image[hi][w-1] - image[hi][w-2]

        # calc for deltaV
    for hi in range(h)[1:h-1]:
        for wi in range(w)[1:w-1]:
            deltaV[hi][wi] = np.sum(np.multiply(image[hi-1:hi+2, wi-1:wi+2], convV))
    for wi in range(w):
        deltaV[0][wi] = image[1][wi] - image[0][wi]
        deltaV[h-1][wi] = image[h-1][wi] - image[h-2][wi]
    for hi in range(h)[1:h-1]:
        deltaV[hi][0] = image[hi+1][0] - image[hi][0]
        deltaV[hi][w-1] = image[hi+1][w-1] - image[hi][w-1]

    deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
    deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

    # calc the theta
    for hi in range(h):
        for wi in range(w):
            if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
                theta[hi][wi] = 0;
            elif(deltaH[hi][wi] == 0):
                theta[hi][wi] = np.pi
            else:
                theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
    theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

    n = 16
    t = 12
    cnt = 0
    hd = np.zeros(n)
    dlen = deltaG_vec.shape[0]
    for ni in range(n):
        for k in range(dlen):
            if((deltaG_vec[k] >= t) and (theta_vec[k] >= (2*ni-1) * np.pi / (2 * n)) and (theta_vec[k] < (2*ni+1) * np.pi / (2 * n))):
                hd[ni] += 1
    hd = hd / np.mean(hd)
    hd_max_index = np.argmax(hd)
    fdir = 0
    for ni in range(n):
        fdir += np.power((ni - hd_max_index), 2) * hd[ni]

    return fdir

#15规整度

def Roughness(path):

    fcrs = Coarseness(path)
    fcon = Contrast(path)
    frgh = fcrs + fcon
    return frgh



#16Chebyshev statistic features
def Chebyshev_Features(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    gfg_matrix = chebyshev.chebvander(img, 3)
    gfg_matrix = sum(gfg_matrix)
    return np.mean(gfg_matrix)


#17Zernike features
def Zernike_Feature(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    img = cv.merge([img, img, img])
    gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    radius = 10
    zernike = mahotas.features.zernike_moments(gray, radius)

    return np.mean(zernike)


#18欧拉数
def Eulernumber(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    eulernumber = skimage.measure.euler_number(img)

    return  eulernumber

#第19特征是Radon transform features
def Radon_Transform_features(path):
    img = io.imread(path, as_gray = grey)
    img = transform.resize(img, (img.shape[0], img.shape[0]))
    theta = np.linspace(0., 180., max(img.shape), endpoint=False)
    sinogram = radon(img, theta=theta)

    return np.mean(sinogram)

#20是Gabor filters
def Gabor_Filters(path):
    img = cv.imread(path, cv.IMREAD_GRAYSCALE)
    retval = cv.getGaborKernel(ksize=(111, 111), sigma=10, theta=60, lambd=10, gamma=1.2)
    # dst	=	cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])
    gabor = cv.filter2D(img, -1, retval)
    return np.mean(gabor)

def storefile(data,fileName):
    with open(fileName,'a',newline='') as f:
        mywriter = csv.writer(f)
        mywriter.writerow(data)






if __name__ == '__main__':
    # 读取文件夹中图片
    Image_dir = 'F:\\Python_Code\\eleven group\\Emotion6\\images\\surprise'  # 添加绝对路径
    Image_glob = os.path.join(Image_dir, "*.jpg")

    Image_name_list = []
    # 将符合条件的 png 文件路径读取到 Image_list 中去
    Image_name_list.extend(glob.glob(Image_glob))
    # print(Image_name_list[::])
    len_Image_name_list = len(Image_name_list)
    #img = cv.imread(Image_name_list[0],cv2.IMREAD_GRAYSCALE)
    #print(img)
    for i in range( len_Image_name_list):
        path = Image_name_list[i]

   # path = 'F:\\Python_Code\\eleven group\\Emotion6\\images\\anger_image\\1.jpg'



        feature = [1 for n in range(21)]
        feature[1] = Meanofimg(path)
        feature[2] = Varianceofimg(path)
        feature[3] = Skew(path)
        feature[4] = Kurtosis(path)
        feature[5] = Contrast(path)
        feature[6] = Energy(path)
        feature[7] = Homogeneity(path)
        feature[8] = Correlation(path)
        feature[9] = Dissimilarity(path)
        feature[10] = Asm(path)
        feature[11] = Entropy(path)
        feature[12] = Coarseness(path)
        feature[13] = Contrast(path)
        feature[14] = Directionality(path)
        feature[15] = Roughness(path)
        feature[16] = Chebyshev_Features(path)
        feature[17] = Zernike_Feature(path)
        feature[18] = Eulernumber(path)
        feature[19] = Radon_Transform_features(path)
        feature[20] = Gabor_Filters(path)
        print(f"It's {i} times: \n")
        print("---------------------")
    #
        storefile(feature,'F:\\Python_Code\\eleven group\\surprise1.csv')
    #    # print(feature)
       #  with open('datalabel.csv', 'a') as f:
       #      csv_writer = csv.writer(f)
       #      for i in range(len(feature)):
       #          csv_writer.writerow(feature[i])


        #      # csv_writer.writerow(["姓名", "年龄", "性别"])

            #f.close()

