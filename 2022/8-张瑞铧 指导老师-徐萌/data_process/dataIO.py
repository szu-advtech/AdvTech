import numpy as np
import cv2
import torch

import matplotlib.pyplot as plt

#############################################################################################################
# s1图像大小为（256，256，2），分别对应图像的长、宽、通道数，通道为【VV,VH】
# s2为清晰图像，云层覆盖率阈值为10%，图像大小为（256，256，13），分别对应图像的长、宽、通道数，数值范围【0，10000】
# s2_cloudy为多云图像，云层覆盖率在20%-70%之间
#############################################################################################################

"""
数据组织格式
0   ['B1': Ultra Blue (Coastal and Aerosol)]
1   ['B2': Blue]
2   ['B3': Green]
3   ['B4': Red]
4   ['B5': Visible and Near Infrared (VNIR)]
5   ['B6': Visible and Near Infrared (VNIR)]
6   ['B7': Visible and Near Infrared (VNIR)]
7   ['B8': Near Infrared (NIR)]
8   ['B8a': Narrow Near Infrared (NIR2)]
9   ['B9': Water Vapour]
10  ['B10': Cirrus]
11  ['B11': Short Wave Infrared (SWIR)]
12  ['B12': Short Wave Infrared (SWIR)]
"""

"""
sensor格式
img_cloudy.shape = (1, 15, 256, 256) 第一位代表训练batch的大小，第二位代表13个sentinel2-cloudy的通道信息 + 2个sentinel1的VV、VH通道
img_predict.shape = (1, 13, 256, 256)
img_true = (1, 13, 256, 256)
"""


# 原始输入数据类型为int16，将其转变为int8类型数据以便matplotlib处理绘制
def convert(img_in):
    img_out = cv2.normalize(img_in, None, 0, 255, cv2.NORM_MINMAX)
    return img_out


def get_RGB_img(red_chnnel, green_chnnel, blue_chnnel):
    img_RGB = np.zeros((256, 256, 3), dtype=np.uint8)
    img_RGB[:, :, 0] = red_chnnel
    img_RGB[:, :, 1] = green_chnnel
    img_RGB[:, :, 2] = blue_chnnel
    return img_RGB


def get_cloudy_true_predict_img(img_cloudy, img_true, img_csm, img_predict, scale=2000):
    # 压缩维度 转NUMPY 转维度 乘以缩放比 再转int8
    # 转换之后维度分别为 15*256*256   13*256*256  13*256*256  1*256*256
    img_cloudy = uint16to8((torch.squeeze(img_cloudy).cpu().numpy() * scale).astype("uint16")).transpose(1, 2, 0)
    img_true = uint16to8((torch.squeeze(img_true).cpu().numpy() * scale).astype("uint16")).transpose(1, 2, 0)
    img_predict = uint16to8((torch.squeeze(img_predict).cpu().numpy() * scale).astype("uint16")).transpose(1, 2, 0)
    img_csm = torch.squeeze(img_csm, dim=0).cpu().numpy().transpose(1, 2, 0)

    # 取RGB
    img_cloudy_RGB = get_RGB_img(img_cloudy[:, :, 3], img_cloudy[:, :, 2], img_cloudy[:, :, 1])
    img_true_RGB = get_RGB_img(img_true[:, :, 3], img_true[:, :, 2], img_true[:, :, 1])
    img_predict_RGB = get_RGB_img(img_predict[:, :, 3], img_predict[:, :, 2], img_predict[:, :, 1])

    plt.subplot(1, 4, 1)
    plt.imshow(img_cloudy_RGB)
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.imshow(img_true_RGB)
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.imshow(img_csm, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.imshow(img_predict_RGB)
    plt.axis('off')

    plt.show()

    """
    output_img[:, 0 * img_size:1 * img_size, :] = img_cld_RGB
    output_img[:, 1 * img_size:2 * img_size, :] = img_fake_RGB
    output_img[:, 2 * img_size:3 * img_size, :] = img_truth_RGB
    output_img[:, 3 * img_size:4 * img_size, :] = img_csm_RGB
    """


"""
from skimage import io
import numpy as np
import cv2

imgpath = 'E:\\mini data\\ROIs1158_spring\\s2_cloudy_1\\ROIs1158_spring_s2_cloudy_1_p30.tif'
img = io.imread(imgpath)

print(img.shape)
"""


def uint16to8(bands, lower_percent=0.001, higher_percent=99.999):
    out = np.zeros_like(bands, dtype=np.uint8)
    n = bands.shape[0]
    for i in range(n):
        a = 0  # np.min(band)
        b = 255  # np.max(band)
        c = np.percentile(bands[i, :, :], lower_percent)
        d = np.percentile(bands[i, :, :], higher_percent)

        t = a + (bands[i, :, :] - c) * (b - a) / (d - c)
        t[t < a] = a
        t[t > b] = b
        out[i, :, :] = t
    return out


#    def validation()
