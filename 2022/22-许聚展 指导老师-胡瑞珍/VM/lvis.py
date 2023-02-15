import json
import numpy as np
import os
from PIL import Image

class Lvis(object):
    def __init__(self, 
        json_path="E:/dataset/lvis/split_train/lvis_common_category.json", 
        image_folders=["E:/dataset/lvis/train2017", "E:/dataset/lvis/val2017", "E:/dataset/lvis/test2017"]):

        self.json_path = json_path
        self.image_folders = image_folders

        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        self.data = data
        self.cat_ids = list(self.data.keys())
        
        self.cat_num = len(self.cat_ids)
        
        names = []
        for cat_id in self.cat_ids:
            cat_name = self.data[cat_id]['name']
            names.append(cat_name)
        self.cat_names = names

    def random_images(self, index, num):
        cat_id = self.cat_ids[index]
        cat_name = self.data[cat_id]['name']
        
        bbox_num = len(self.data[cat_id]['bboxes'])
        bbox_list = [i for i in range(bbox_num)]

        if bbox_num < num:
            num = bbox_num

        sample_ids = np.random.choice(bbox_list, num, replace=False)

        images = []

        for sample_id in sample_ids:
            image_id = self.data[cat_id]['images'][sample_id]
            x,y,w,h = self.data[cat_id]['bboxes'][sample_id]
            
            for folder in self.image_folders:
                image_path = os.path.join( folder, "%012d.jpg" % image_id )
                if os.path.exists(image_path):
                    break
            img = Image.open( image_path )
            # crop_img = img.crop([x, y, x+w, y+h])
            # images.append(crop_img)
            images.append(img)

        return images, cat_name
    
    def random_test(self, task_num, img_num_pre_task):
        index_list = [ i for i in range(self.cat_num) ]

        for i in range(task_num):
            source_list = []
            target_list = []
            label_list = []
            
            sample_ids = np.random.choice(index_list, img_num_pre_task, replace=False)
            for cat_id in sample_ids:
                images, cat_name = self.random_images(cat_id, 2)

                source_list.append(images[0])
                target_list.append(images[1])
                label_list.append(cat_name)
            
            yield source_list, target_list, label_list


if __name__ == '__main__':
    
    dataset = Lvis()

    np.random.seed(6)
    
    images, cat_name = dataset.random_images(9, 2)

    a = 1