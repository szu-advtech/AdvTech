import json
import numpy as np
import os
from PIL import Image

class Lvis(object):
    def __init__(self, json_path, image_folders) -> None:
        self.json_path = json_path
        self.image_folders = image_folders

        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        self.data = data
        self.cat_ids = list(self.data.keys())
        
        self.cat_num = len(self.cat_ids)
        print(self.cat_num)

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
            crop_img = img.crop([x, y, x+w, y+h])
            images.append(crop_img)
        
        return images, cat_name
    

if __name__ == '__main__':
    
    json_path = "./split_val/lvis_common_category.json"
    image_folders = ["./train2017", "./val2017", "./test2017"]

    dataset = Lvis(json_path, image_folders)

    np.random.seed(6)
    
    images, cat_name = dataset.random_images(3, 2)
    print(cat_name)
    a = 1