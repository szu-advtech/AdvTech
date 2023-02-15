import numpy as np
import cv2
import os
 
# SSR
def replaceZeroes(data):
    min_nonzero = min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data
 
def SSR(src_img, size):
    L_blur = cv2.GaussianBlur(src_img, (size, size), 0)
    img = replaceZeroes(src_img)
    L_blur = replaceZeroes(L_blur)
 
    dst_Img = cv2.log(img/255.0)
    dst_Lblur = cv2.log(L_blur/255.0)
    dst_IxL = cv2.multiply(dst_Img,dst_Lblur)
    log_R = cv2.subtract(dst_Img, dst_IxL)
 
    dst_R = cv2.normalize(log_R,None,0,255,cv2.NORM_MINMAX)
    log_uint8 = cv2.convertScaleAbs(dst_R)
    return log_uint8


if __name__ == '__main__':
    old = ""
    new = ""
    f = os.listdir(old)
    for filename in f:
        img_file = old+'/'+filename
        print(img_file)
        size = 3
        src_img = cv2.imread(img_file)
        b_gray, g_gray, r_gray = cv2.split(src_img)
        b_gray = SSR(b_gray, size)
        g_gray = SSR(g_gray, size)
        r_gray = SSR(r_gray, size)
        result = cv2.merge([b_gray, g_gray, r_gray])
        cv2.imwrite(new+'/'+filename, result)