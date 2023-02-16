import numpy as np
import expression
from detect import face_rec
from evalution import evalute
import cv2


# 样本数据保存
def save_data(score_list):
    score_list = np.array(score_list)
    np.savetxt("score_list.txt", score_list, fmt="%.15f")


# 画框、输入文字
def design(draw, rectangles, exp, emotion_score):
    font = cv2.FONT_HERSHEY_SIMPLEX

    for (left, top, right, bottom) in rectangles:
        cv2.rectangle(draw, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(draw, '{} {:.3f}'.format(exp, emotion_score[0][0]), (left, bottom), font, 1, (255, 255, 255), 2)

        top2 = int((bottom - top) / 3 + top)
        bottom2 = int(bottom - (bottom - top) / 3)
        cv2.rectangle(draw, (left, top2), (right, bottom2), (255, 255, 255), -1)

    return draw


def main():
    detect = face_rec()
    video_capture = cv2.VideoCapture(0)      # 打开摄像头
    num = 0     # 人脸区域获取计数
    score_list = []

    while True:
        ret, draw = video_capture.read()    # 视频采集
        imageSave, rectangles = detect.recognize(draw, num)   # 人脸检测
        exp, score = expression.facial_expression(imageSave)    # 表情分析
        emotion_score = evalute(score)     # 教学评价

        design(draw, rectangles, exp, emotion_score)    # 画框、输出得分
        score_list.append(score)       # 采集情感得分样本数据

        cv2.imshow('Video', draw)
        # 按‘q’退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            # save_data(score_list)   # 保存样本数据
            break
        num += 1

    # 释放摄像头、关闭所有窗口
    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
