import json
import cv2
import matplotlib.pyplot as plt
import numpy as np

"""
使用opencv进行多边形填充来生成mask
""" 

img = cv2.imread(
    "D:/_PensonalFiles/szu/RS/solar_pv/pv_data/Duke/TIF_dataset/Oxnard/619851917.tif")
#BGR->RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask = np.zeros_like(img)
 
json_file = {}
with open("D:/_PensonalFiles/szu/RS/solar_pv/pv_data/Duke/Duke_PV_dataset/SolarArrayPolygons.json", "r") as f:
    json_file = f.read()

json_file = json.loads(json_file)

polygons = json_file["polygons"]
#print(len(points))
i = 1
image_name = ''
for i in range(len(polygons)):
    if(polygons[i]['image_name'] == '619851917'):
        #image_name = '619851905'
        tmp = np.array(polygons[i]['polygon_vertices_pixels'], np.int32)
        cv2.fillPoly(mask, [tmp], (255, 255, 255))
#print(points[0])
#points = np.array(points, np.int32)



#cv2.rectangle(json_file,(box[0][0], box[0][1]), (box[1][0], box[1][1]) ,(125,125,125),2)
#cv2.polylines(img, [points], 1, (0,0,255))
#cv2.fillPoly(mask, [tmp], (255, 255, 255))
img_add = cv2.addWeighted(mask, 0.3,img,0.7,0)
cv2.imwrite(
    "D:/_PensonalFiles/szu/RS/solar_pv/pv_data/Duke/TIF_dataset/Oxnard/label/619851917_mask.png", mask)
cv2.imwrite("D:/_PensonalFiles/szu/RS/solar_pv/pv_data/Duke/TIF_dataset/Oxnard/label/619851917_mask.tif", mask)
print('tif generate successful')
plt.imshow(mask)
plt.show()