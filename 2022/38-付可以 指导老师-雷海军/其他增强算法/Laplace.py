import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
 
def laplacian(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    image_lap = cv2.filter2D(image, cv2.CV_8UC3, kernel)
   # cv2.imwrite('th1.jpg', image_lap)
    return image_lap

if __name__ == '__main__':
    old = ""
    new = ""
    f = os.listdir(old)
    for filename in f:
        img_file = old+'/'+filename
        print(img_file)
        img = cv2.imread(img_file)
        final = laplacian(img)
        cv2.imwrite(new+'/'+filename, final)