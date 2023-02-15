import random

import torch
import os
import numpy as np
import cv2
from PIL import Image
from torch.utils import data
from torchvision.transforms import transforms


def get_raf(train_root, train_file_list, test_root, test_file_list,transform_train=None, transform_val=None):

    # load data
    triplet_train_dataset = triplet_Dataset_train(train_root, train_file_list, transform=transform_train)

    train_dataset = Dataset_train(train_root, train_file_list, transform=transform_train)

    test_dataset = Dataset_test(test_root, test_file_list, transform=transform_val)

    print (f"#train: {len(train_dataset)} #test: {len(test_dataset)}")
    return train_dataset,triplet_train_dataset,test_dataset

def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = Image.fromarray(img)
            return img
    except IOError:
        print('Cannot load image ' + path)

# random sampling
class triplet_Dataset_train(torch.utils.data.Dataset):
    def __init__(self,root, file_list,transform=None, loader=img_loader):
        self.root = root
        self.transform = transform
        self.loader = loader

        '''
            # Basic Notes:
            # 0: Surprised
            # 1: Fear
            # 2: Disgust
            # 3: Happy
            # 4: Sad
            # 5: Angry
            # 6: Neutral
        '''

        # Classified storage of data
        image_list_0 = []
        image_list_1 = []
        image_list_2 = []
        image_list_3 = []
        image_list_4 = []
        image_list_5 = []
        image_list_6 = []

        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:

            image_path, label_name= info.split(' ')
            if int(label_name) == 0:
                image_list_0.append(image_path)
            elif int(label_name) == 1:
                image_list_1.append(image_path)
            elif int(label_name) == 2:
                image_list_2.append(image_path)
            elif int(label_name) == 3:
                image_list_3.append(image_path)
            elif int(label_name) == 4:
                image_list_4.append(image_path)
            elif int(label_name) == 5:
                image_list_5.append(image_path)
            elif int(label_name) == 6:
                image_list_6.append(image_path)

        self.image_list_0 = image_list_0
        self.image_list_1 = image_list_1
        self.image_list_2 = image_list_2
        self.image_list_3 = image_list_3
        self.image_list_4 = image_list_4
        self.image_list_5 = image_list_5
        self.image_list_6 = image_list_6
        self.class_nums = 7

    def __getitem__(self, index):

        label = index % 7
        if label == 0:
            img_path = self.image_list_0[random.randint(0,len(self.image_list_0)-1)]

        elif label == 1:
            img_path = self.image_list_1[random.randint(0,len(self.image_list_1)-1)]

        elif label == 2:
            img_path = self.image_list_2[random.randint(0,len(self.image_list_2)-1)]

        elif label == 3:
            img_path = self.image_list_3[random.randint(0,len(self.image_list_3)-1)]

        elif label == 4:
            img_path = self.image_list_4[random.randint(0,len(self.image_list_4)-1)]

        elif label == 5:
            img_path = self.image_list_5[random.randint(0,len(self.image_list_5)-1)]

        elif label == 6:
            img_path = self.image_list_6[random.randint(0,len(self.image_list_6)-1)]


        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_list_0) + len(self.image_list_1) + len(self.image_list_2) + len(self.image_list_3) + \
               len(self.image_list_4) + len(self.image_list_5) + len(self.image_list_6)

class Dataset_train(torch.utils.data.Dataset):
    def __init__(self,root,file_list,transform=None, loader=img_loader):
        super(Dataset_train, self).__init__()
        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []

        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_path, label_name = info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        # 获得类别数
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_list)

class Dataset_test(torch.utils.data.Dataset):
    def __init__(self, root, file_list, transform=None, loader=img_loader):
        super(Dataset_test, self).__init__()

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []

        with open(file_list) as f:
            img_label_list = f.read().splitlines()
        for info in img_label_list:
            image_path, label_name= info.split(' ')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(os.path.join(self.root, img_path))
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_list)





