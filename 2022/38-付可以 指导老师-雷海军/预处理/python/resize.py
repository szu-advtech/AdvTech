import numpy as np
import cv2 as cv
from PIL import Image
import os

# 对图片尺寸进行修改
def resizeChange(img):
    print(img.size)

    new_image = img.resize((112,112))
    print(new_image.size)
    return new_image

# 将img1的尺寸改为img2的尺寸
def resizeRechange(img1,img2):
    print(img1.size)
    print(img2.size)

    final = img1.resize((img2.size))
    return final

# 对文件夹path_dir1的图片尺寸进行修改并保存到path_dir2种
def ChangeForMany(path_dir1,path_dir2):
    f = os.listdir(path_dir1)
    for filename in f:
        img_file = path_dir1+'/'+filename
        print(img_file)
        img = Image.open(img_file)
        m = resizeChange(img)
        m.save(path_dir2+'/'+filename)

# 将文件夹path_dir1的图片尺寸修改位文件夹path_dir2对应图片的尺寸，并保存到path_dir3种
def ReChangeForMany(path_dir1,path_dir2,path_dir3):
    f = os.listdir(path_dir1)
    q = os.listdir(path_dir2)
    for filename in f:
        img_file1 = path_dir1+'/'+filename
        print(img_file1)
        img1 = Image.open(img_file1)

        img_file2 = path_dir2+'/'+filename
        print(img_file2)
        img2 = Image.open(img_file2)


        m = resizeRechange(img1,img2)
        m.save(path_dir3+'/'+filename)

ChangeForMany("input_train","input_train")
ChangeForMany("input_wb_train","input_wb_train")
ChangeForMany("input_ce_train","input_ce_train")
ChangeForMany("input_gc_train","input_gc_train")
ChangeForMany("gt_train","gt_train")
# ReChangeForMany("test_igs","old","final_resize")