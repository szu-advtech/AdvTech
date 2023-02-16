import os
import shutil
import skimage.util as util
from skimage import io
from skimage.transform import resize

root = "./dataset/deepfashionHD/"
root1 = "./data/deepfashionHD/"
if __name__ == '__main__':
    with open('./data/train.txt','r') as fd:
        image_path_list = fd.readlines()

    for image_path in image_path_list:
        image_path = image_path.replace("\n","")
        head,tail = os.path.split(image_path)
        head = root + head.replace("img","train_img")
        if not os.path.exists(head):
            os.makedirs(head)
        source_path = root1 + image_path
        source_path = source_path.replace("/","\\")
        if not os.path.exists(source_path):
            print("error~~~~~")
        target_path = head +"\\"+tail
        target_path = target_path.replace("/","\\")
        shutil.copyfile(source_path,target_path)