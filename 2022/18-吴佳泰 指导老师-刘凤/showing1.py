import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
for _data_name in ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']:
    data_path1 = './data/TestDataset/{}/images/'.format(_data_name)
    data_path2 = './data/TestDataset/{}/masks/'.format(_data_name)
    data_path3 ='./results/PraNet/{}/'.format(_data_name)
    data_path4 ='./results/UNet/{}/'.format(_data_name)
    data_path5 ='./results/UNet++/{}/'.format(_data_name)
    for i in range(149,209):
        img1 = Image.open(data_path1 + '{}.png'.format(i))
        img2 = Image.open(data_path2 + '{}.png'.format(i))
        img3 = Image.open(data_path3 + '{}.png'.format(i))
        img4 = Image.open(data_path4 + '{}.png'.format(i))
        img5 = Image.open(data_path5 + '{}.png'.format(i))
        plt.figure(figsize=(50,5)) #设置窗口大小
        plt.suptitle('结果对比图') # 图片名称
        plt.subplot(1,5,1), plt.title('image')
        plt.imshow(img1), plt.axis('off')
        plt.subplot(1,5,2), plt.title('GroundTruth')
        plt.imshow(img2), plt.axis('off')
        plt.subplot(1,5,3), plt.title('PraNet')
        plt.imshow(img3), plt.axis('off')
        plt.subplot(1,5,4), plt.title('UNet')
        plt.imshow(img4), plt.axis('off')
        plt.subplot(1,5,5), plt.title('UNet++')
        plt.imshow(img5), plt.axis('off')
        plt.ion()
        plt.pause(3)
        plt.close()
