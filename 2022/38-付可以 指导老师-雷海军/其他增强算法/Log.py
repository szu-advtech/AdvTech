import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
 
def log(image):
    image_log = np.uint8(np.log(np.array(image) + 1))
    cv2.normalize(image_log, image_log, 0, 255, cv2.NORM_MINMAX)
    # 转换成8bit图像显示
    cv2.convertScaleAbs(image_log, image_log)
    return image_log

if __name__ == '__main__':
    img_file = 'Retinex/old/h-500.png'
    img = cv2.imread(img_file)
    final = log(img)
    cv2.imwrite('./test-opencv-log.png',final)

if __name__ == '__main__':
    old = ""
    new = ""
    f = os.listdir(old)
    for filename in f:
        img_file = old+'/'+filename
        print(img_file)
        img = cv2.imread(img_file)
        final = log(img)
        cv2.imwrite(new+'/'+filename, final)