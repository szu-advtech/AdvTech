import cv2 as cv
 
def ORB_Feature(img1, img2):
 
    # 初始化ORB
    orb = cv.ORB_create(1000)
 
    # 寻找关键点
    kp1 = orb.detect(img1)
    kp2 = orb.detect(img2)
    print(len(kp1))
    print(len(kp2))

    # 计算描述符
    kp1, des1 = orb.compute(img1, kp1)
    kp2, des2 = orb.compute(img2, kp2)
 
    # 画出关键点
    outimg1 = cv.drawKeypoints(img1, keypoints=kp1, outImage=None)
    outimg2 = cv.drawKeypoints(img2, keypoints=kp2, outImage=None)
   
    # 显示关键点
    import numpy as np
    outimg3 = np.hstack([outimg1, outimg2])
    cv.imshow("Key Points", outimg3)
    cv.waitKey(0)
 
    # 初始化 BFMatcher
    bf = cv.BFMatcher(cv.NORM_HAMMING)
 
    # 对描述子进行匹配
    matches = bf.match(des1, des2)
 
    # 计算最大距离和最小距离
    min_distance = matches[0].distance
    max_distance = matches[0].distance
    for x in matches:
        if x.distance < min_distance:
            min_distance = x.distance
        if x.distance > max_distance:
            max_distance = x.distance
 
    # 筛选匹配点
    '''
        当描述子之间的距离大于两倍的最小距离时，认为匹配有误。
        但有时候最小距离会非常小，所以设置一个经验值30作为下限。
    '''
    good_match = []
    for x in matches:
        if x.distance <= max(2 * min_distance, 30):
            good_match.append(x)
 
    # 绘制匹配结果
    draw_match(img1, img2, kp1, kp2, good_match)
    print(len(good_match))
 
def draw_match(img1, img2, kp1, kp2, match):
    outimage = cv.drawMatches(img1, kp1, img2, kp2, match, outImg=None)
    cv.imshow("Match Result", outimage)
    cv.waitKey(0)
 
if __name__ == '__main__':
    # 读取图片
    # image1 = cv.imread('Histogram-datasets/orb-test/bg_15_out.png')
    # image2 = cv.imread('Histogram-datasets/orb-test/bg_16_out.png')

    # 原图与其经过增强后特征点的比较
    image1 = cv.imread('C:/Users/Administrator/Desktop/test/Water-Net_Code-master/old/600_img_.png')
    image2 = cv.imread('C:/Users/Administrator/Desktop/test/Water-Net_Code-master/final_resize/600_img_.png')
    ORB_Feature(image1, image2)