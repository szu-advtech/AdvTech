import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# #用直方图分别统计图像各个像素三个颜色通道数值数量的分布
# img = cv2.imread('/content/Rust_032.jpg')
# color = ('blue','green','red')
# #enumerate函数第一个返回索引index，第二个返回元素element
# for i,color in enumerate(color):
#     #参数说明：
#     #一、images（输入图像）参数必须用方括号括起来。
#     #二、计算直方图的通道。
#     #三、Mask（掩膜），一般用None，表示处理整幅图像。
#     #四、histSize，表示这个直方图分成多少份（即多少个直方柱）。
#     #五、range，直方图中各个像素的值，[0.0, 256.0]表示直方图能表示像素值从0.0到256的像素。
#     hist = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(hist,color = color)
#     plt.xlim([0,256])
# plt.show()
# cv2.waitKey()

# 自适应直方图均衡化(AHE)
from skimage import exposure
file_to_open = 'test_real/600_img_.png'
img2 = cv2.imread(file_to_open)
plt.figure()
plt.imshow(img2)
plt.show()

img = exposure.equalize_adapthist(img2)
im = Image.fromarray(np.uint8(img * 255))
im.save('AHE.png')
plt.figure()
plt.imshow(img)
plt.show()
