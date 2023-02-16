import cv2
import math
import numpy as np
import os

def FKY_UCIQE(image):
    # 先将rgb空间转换到hsv空间——色调（H）、饱和度（S）和明度（V）
    hsv = cv2.cvtColor(image,cv2.COLOR_RGB2HSV)
    H,S,V = cv2.split(hsv)

    # 计算色调H的标准差σc
    delta = np.std(H) /180
    # 计算色度的饱和度S平均值μs
    mu = np.mean(S) /255

    # 计算亮度对比度的平均值conl
    n,m = np.shape(V)
    # 所需计算像素的个数
    number = math.floor(n*m/100)
    Maxsum,Minsum = 0,0
    V1,V2 = V / 255,V / 255
    
    for i in range(1,number+1):
        Maxvalue = np.amax(np.amax(V1))
        x,y = np.where(V1 == Maxvalue)
        Maxsum = Maxsum + V1[x[0],y[0]]
        V1[x[0],y[0]] = 0
    
    top = Maxsum/number
    
    for i in range(1, number+1):
        Minvalue = np.amin(np.amin(V2))
        X,Y = np.where(V2 == Minvalue)
        Minsum = Minsum + V2[X[0],Y[0]]
        V2[X[0],Y[0]] = 1
    
    bottom = Minsum/number
    
    conl = top-bottom
    # print("-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=")
    # print(conl)
    return 0.25*delta + 0.6*conl + 0.15*mu

# print(uciqe(cv2.imread("test_real/600_img_.png")))
# print(uciqe(cv2.imread("HE-1.png")))
# print(uciqe(cv2.imread("HE-2.png")))
# print(uciqe(cv2.imread("AHE.png")))
# print(uciqe(cv2.imread("CLAHE-1.png")))
# print(uciqe(cv2.imread("CLAHE-2.png")))


if __name__== '__main__':
    file_path = ""
    file_num = len(os.listdir(file_path))
    sum = 0

    f = os.listdir(file_path)
    # 遍历所有图像，对每一张图像进行评估，并计算所有图像UCIQE评分的平均值
    for filename in f:
        img_file = file_path+'/'+filename
        print(img_file)
        img = cv2.imread(img_file)
        final = FKY_UCIQE(img)
        print(final)
        sum = sum + final
    print(sum/file_num)