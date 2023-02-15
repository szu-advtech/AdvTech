import cv2
import numpy as np
import os

def normalize_transform(gray_img):
    '''
    :param gray_img:
    :return:
    '''
    Imin, Imax = cv2.minMaxLoc(gray_img)[:2]
    Omin, Omax = 0, 255
    # 计算a和b的值
    a = float(Omax - Omin) / (Imax - Imin)
    b = Omin - a * Imin
    out = a * gray_img + b
    out = out.astype(np.uint8)
    return out

if __name__== '__main__':
    old = ""
    new = ""
    f = os.listdir(old)
    for filename in f:
        img_file = old+'/'+filename
        print(img_file)
        img = cv2.imread(img_file)

        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        b_out = normalize_transform(b)
        g_out = normalize_transform(g)
        r_out = normalize_transform(r)
        nor_out = np.stack((b_out, g_out, r_out), axis=-1)
        cv2.imwrite(new+'/'+filename, nor_out)