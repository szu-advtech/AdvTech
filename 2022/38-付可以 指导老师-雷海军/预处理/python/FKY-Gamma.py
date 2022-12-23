import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
 
def gamma(image):
    fgamma = 2
    # fgamma = 0.7
    image_gamma = np.uint8(np.power((np.array(image) / 255.0), fgamma) * 255.0)
    cv2.normalize(image_gamma, image_gamma, 0, 255, cv2.NORM_MINMAX)
    cv2.convertScaleAbs(image_gamma, image_gamma)
    return image_gamma

if __name__ == '__main__':
    old = ""
    new = ""
    f = os.listdir(old)
    for filename in f:
        img_file = old+'/'+filename
        print(img_file)
        img = cv2.imread(img_file)
        final = gamma(img)
        cv2.imwrite(new+'/'+filename, final)