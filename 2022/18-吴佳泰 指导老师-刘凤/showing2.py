import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
data_path1 = './data/Test_Data/Testing_data/'
data_path2 = './data/Test_Data/GroundTruth/'
data_path3 ='./results/result/'
for filename in os.listdir(data_path1):
    img1 = Image.open(data_path1 + filename)
    filename=filename.replace(".jpg",'')
    img2 = Image.open(data_path2 + filename+"_Segmentation.png")
    img3 = Image.open(data_path3 + filename+".png")
    plt.figure(figsize=(50,5)) #设置窗口大小
    plt.suptitle('结果对比图') # 图片名称
    plt.subplot(1,5,1), plt.title('image')
    plt.imshow(img1), plt.axis('off')
    plt.subplot(1,5,2), plt.title('GroundTruth')
    plt.imshow(img2), plt.axis('off')
    plt.subplot(1,5,3), plt.title('PraNet')
    plt.imshow(img3), plt.axis('off')
    plt.ion()
    plt.pause(3)
    plt.close()
