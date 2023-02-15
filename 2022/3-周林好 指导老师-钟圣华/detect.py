import os
import cv2
import numpy as np
from net.mtcnn import mtcnn

'''
人脸检测模块
'''

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class face_rec:
    """
    创建mtcnn的模型
    """
    def __init__(self):
        self.mtcnn_model = mtcnn()
        self.threshold = [0.5, 0.6, 0.7]

    """
    检测人脸
    """
    def recognize(self, draw, num):
        height, width, _ = np.shape(draw)
        draw_rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

        rectangles = self.mtcnn_model.detectFace(draw_rgb, self.threshold)
        if len(rectangles) == 0:
            return
        rectangles = np.array(rectangles, dtype=np.int32)

        rectangles = rectangles[:, 0:4]

        # 获取人脸
        imageSave = ''
        for rectangle in rectangles:
            X = int(rectangle[0])
            Y = int(rectangle[1])
            W = int(rectangle[2]) - int(rectangle[0])
            H = int(rectangle[3]) - int(rectangle[1])
            imageSave = draw_rgb[Y:Y + H, X:X + W]
            # image_name = '%s%d.jpg' % ('test/', num)      # 获取每帧检测出的人脸区域
            # cv2.imwrite(image_name, imageSave)

        height, width, _ = np.shape(imageSave)

        return imageSave, rectangles
