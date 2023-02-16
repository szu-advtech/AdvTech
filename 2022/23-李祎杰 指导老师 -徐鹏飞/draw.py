import json
import math

import cv2
import numpy as np

#data_path='./results/Example1_16.json'
data_path='Example1.json'

with open(data_path, 'r') as data_file:
    data = json.load(data_file)
    print(json.dumps(data, indent=2))

layouts = data['layouts']
elements = layouts[0]['elements']
canvas_w=layouts[0]['canvasWidth']
canvas_h=layouts[0]['canvasHeight']

canvas = np.zeros((canvas_h,canvas_w,3),np.uint8)
img=canvas+255

font = cv2.FONT_HERSHEY_TRIPLEX
point_color = (193,255,193) # BGR
thickness = -1
lineType = 4

for i in range(len(elements)):
    ptLeftTop = (int(elements[i]['x']), int(elements[i]['y']))
    ptRightBottom = (int(elements[i]['x']+elements[i]['width']), int(elements[i]['y']+elements[i]['height']))
    if elements[i]['type']=="text":
        point_color = (255,228,181)
    else:
        point_color = (193, 255, 193)
    #cv2.putText(img, str(i), ptLeftTop, math.ceil(font), 1, (0, 0, 0), 1)
    cv2.rectangle(img, ptLeftTop, ptRightBottom, point_color, thickness, lineType)
    if elements[i]['type']=="text":
        # cv2.putText(img, str,origin,font,size,color,thickness)
        textpos = [ptLeftTop[0] + int(elements[i]['width']/2), ptLeftTop[1] + int(elements[i]['height']/2)]
        cv2.putText(img, 'text', textpos, math.ceil(font), 1, (0, 0, 0), 1)


cv2.namedWindow("Draw Layouts")
cv2.imshow('Draw Layouts', img)
cv2.waitKey (0)
cv2.destroyAllWindows()

print("test")