import numpy as np
import cv2
import dlib
from collections import OrderedDict
import os
def shape_to_np(shape, dtype="int"):
    #print(shape.num_parts)
    coords=np.zeros((shape.num_parts,2),dtype=dtype)
    for i in range(0,shape.num_parts):
        coords[i]=(shape.part(i).x,shape.part(i).y)
    return coords
FACIAL_LANDMARKS_68_IDXS=OrderedDict([("mouth",(48,68)),("right_eyebrow",(17,22)),("left_eyebrow",(22,27)),("right_eye",(36,42))
                                      ,("left_eye",(42,48)),("nose",(27,36)),("jaw",(0,17))])
detector=dlib.get_frontal_face_detector()
predictor=dlib.shape_predictor('model/shape_predictor_68_face_landmarks.dat')

#os.makedirs('{:s}'.format("masks2"), exist_ok=True)
imgpath="./test_data/test_image"
maskpath="./test_data/test_mask"
names=os.listdir(imgpath)
for name in names:
    imgName=os.path.join(imgpath,name)
    img=cv2.imread(imgName)
    mask=np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    rects=detector(img,1)
    for (i,rect) in enumerate(rects):
        shape=predictor(img,rect)
        print(shape)
        shape=shape_to_np(shape)

        (j,k)=FACIAL_LANDMARKS_68_IDXS["mouth"]
        pts=shape[j:k]
        #mask=np.zeros(img.shape[:2],dtype=np.uint8)
        #(x,y,w,h)=cv2.boundingRect(pts)
        #roi=np.array([[x,y],[x+w,y],[x+w,y+h],[x,y+h]],np.int32)
        cv2.fillConvexPoly(mask,pts,(255,255,255))
        #cv2.imshow("mask1",mask)
        savepth=os.path.join(maskpath,name[:-4]+".png")
        cv2.imwrite(savepth,mask)
    """alpha=1
    beta=1
    gamma=0
    mask_image=cv2.addWeighted(img,alpha,mask,beta,gamma)
    cv2.imshow("mask_img1",mask_image)
    cv2.imwrite("mask_image1.png",mask_image)
    cv2.waitKey(0)"""

