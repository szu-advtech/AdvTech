import pickle
from skimage import transform
from torchvision import transforms
import os
import numpy as np
import skimage
from skimage import io
import torch.utils.data as data
import random
train_label_dict={}
valid_label_dict={}
test_label_dict={}
pretrain_label_dict={}
metatrain_label_dict={}

def read_label_set(type):
    import pandas as pd
    if type=='train':
        path = 'splits/train.csv'
    elif type=='valid':
        path = 'splits/val.csv'
    elif type=='test':
        path = 'splits/test5_1.csv'
    elif type=='pretrain':
        path = 'splits/pretrain.csv'
    elif type=='metatrain':
        path = 'splits/metatrain.csv'
                        
    data = pd.read_csv(path)
    ylabel = np.array(data[['label']]).tolist()
    if type=='train':
        for y in ylabel:
            y_idx = y[0]
            if y_idx not in train_label_dict:
                train_label_dict[y_idx] = "1"
    elif type=='valid':
        for y in ylabel:
            y_idx = y[0]
            if y_idx not in valid_label_dict:
                valid_label_dict[y_idx] = "1"
    elif type=='test':
        for y in ylabel:
            y_idx = y[0]
            if y_idx not in test_label_dict:
                test_label_dict[y_idx] = "1"
    elif type=='pretrain':
        for y in ylabel:
            y_idx = y[0]
            if y_idx not in pretrain_label_dict:
                pretrain_label_dict[y_idx] = "1"
    elif type=='metatrain':
        for y in ylabel:
            y_idx = y[0]
            if y_idx not in metatrain_label_dict:
                metatrain_label_dict[y_idx] = "1"                

read_label_set("train")
print(train_label_dict)
read_label_set("valid")
print(valid_label_dict)
read_label_set("test")
print(test_label_dict)
read_label_set("pretrain")
print(pretrain_label_dict)
read_label_set("metatrain")
print(metatrain_label_dict)

class image_dataset(data.Dataset):
    def __init__(self,
                 root,
                 transform=None
                 ):
        self.root = root
        self.transform = transform
        img_data, class_dict, img_user, img_task = self.load_data()
        self.img_data = img_data
        self.class_dict = class_dict
        self.img_user = img_user
        self.img_task = img_task

    def __getitem__(self, idx):
        item_img_path = self.img_data[idx]
        y = self.class_dict[idx]
        x = io.imread(item_img_path)
        x = skimage.img_as_ubyte(x)
        x = np.delete(x, -1, axis=2)
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.img_data)

    def load_data(self):
        img_data = []
        class_dict = {}
        img_user = {}
        img_task = {}
        i = 0
        print(self.root)
        for root, dirs, files in os.walk(self.root):
            #print(files)
            for file in files:
                #print(file)
                file_name = file.split('.')#n0322051300000158.jpg
                label = file_name[0][0:9]
                task_key = "miniImagenet"
                if label not in test_label_dict:#train/valid/test
                    continue
                
                path = os.path.join(root, file)
                image_data = io.imread(path)
                image_data=transform.resize(image_data,(224, 224))#change 224*224
                image_data = skimage.img_as_ubyte(image_data)
                #image_data = np.delete(image_data, -1, axis=2)
                img_data.append(image_data)
                cd = label
                if cd not in class_dict.keys():
                    class_dict[cd] = []
                    class_dict[cd].append(i)
                else:
                    class_dict[cd].append(i)
                fn = file_name[0][0:1]
                if fn not in img_user.keys():
                    img_user[fn] = []
                    img_user[fn].append(i)
                else:
                    img_user[fn].append(i)
                if task_key not in img_task.keys():
                    img_task[task_key] = []
                    img_task[task_key].append(i)
                else:
                    img_task[task_key].append(i)

                i = i + 1
        img_data = np.array(img_data)
        return img_data, class_dict, img_user, img_task

def get_data(way=5, seed=42):
    random.seed(seed)
    test_data_set = image_dataset("data", transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]))
    save_data = {}
    save_data["image_data"] = test_data_set.img_data
    #print(save_data["image_data"])
    save_data["class_dict"] = test_data_set.class_dict
    print(save_data["class_dict"])
    save_data["img_user"] = test_data_set.img_user
    print(save_data["img_user"])
    save_data["img_task"] = test_data_set.img_task
    print(save_data["img_task"])
    return save_data
