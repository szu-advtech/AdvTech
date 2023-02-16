import cv2
import os

img_path = "C:\\Users\\James\\Desktop\\img"
label_path = "C:\\Users\\James\\Desktop\\label"
save_path = "C:\\Users\\James\\Desktop\\img"

labelList = os.listdir(label_path)

for i in range(len(labelList)):
    img = cv2.imread(label_path + os.sep + labelList[i])
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(save_path + os.sep + f'{i}.png', gray)
