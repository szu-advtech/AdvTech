import cv2
import math
import numpy as np
 
# 值越高，表示图像质量越好
image = cv2.imread("new.png")#图片路径
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)  # RGB转为HSV 
H, S, V = cv2.split(hsv)
delta = np.std(H) /180  #色度的标准差 
mu = np.mean(S) /255  #饱和度的平均值 
n, m = np.shape(V)
number = math.floor(n*m/100)  #所需像素的个数  
Maxsum, Minsum = 0, 0
V1, V2 = V /255, V/255
 
for i in range(1, number+1):
    Maxvalue = np.amax(np.amax(V1))
    x, y = np.where(V1 == Maxvalue)
    Maxsum = Maxsum + V1[x[0],y[0]]
    V1[x[0],y[0]] = 0
 
top = Maxsum/number
 
for i in range(1, number+1):
    Minvalue = np.amin(np.amin(V2))
    X, Y = np.where(V2 == Minvalue)
    Minsum = Minsum + V2[X[0],Y[0]]
    V2[X[0],Y[0]] = 1
 
bottom = Minsum/number
 
conl = top-bottom
 ###对比度 
uciqe = 0.4680*delta + 0.2745*conl + 0.2575*mu
print(delta, conl, mu)
print(uciqe)
