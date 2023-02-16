import cv2
import numpy as np
def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image-a_min)/(a_max - a_min)

def cvt2heatmap(gray):
    # uni8表示的是无符号整形，表示范围是[0.255]的整数，cv2.COLORMAP_JET用于生成热力图
    heatmap = cv2.applyColorMap(np.uint8(gray), cv2.COLORMAP_JET)
    return heatmap