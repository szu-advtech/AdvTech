import os
import cv2
 
# 彩色图像全局直方图均衡化
def FKY_hisEqulColorAll(img):
    # 将RGB图像转换到YCrCb空间中
    ycrcb = cv2.cv2tColor(img, cv2.COLOR_BGR2YCR_CB)
    # 将YCrCb图像通道分离
    channels = cv2.split(ycrcb)
    # 对第1个通道即亮度通道进行全局直方图均衡化并保存
    cv2.equalizeHist(channels[0], channels[0])
    # 将处理后的通道和没有处理的两个通道合并，命名为ycrcb
    cv2.merge(channels, ycrcb)
    # 将YCrCb图像转换回RGB图像
    cv2.cv2tColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img)
    return img

if __name__ == '__main__':
    old = ""
    new = ""
    f = os.listdir(old)
    for filename in f:
        img_file = old+'/'+filename
        print(img_file)
        img = cv2.imread(img_file)
        final = FKY_hisEqulColorAll(img)
        cv2.imwrite(new+'/'+filename, final)