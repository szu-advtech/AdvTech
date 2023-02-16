import os

import numpy as np
import skimage
from skimage import io
import torch.utils.data as data

default_digits = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
default_letters = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
                   'U', 'V', 'W', 'X', 'Y', 'Z'}
default_char_set = set.union(default_digits, default_letters)
default_people_set = {"p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"}
default_phone_set = {"sss", "xiaomi"}
default_angle_set = {"0", "30", "45", "60", "90", "120", "135", "150"}
default_env_set = {"lab", "out", "loud"}
default_hand_set = {"hand", "none"}
letter_dict = {'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
               'U', 'V', 'W', 'X', 'Y', 'Z'}



def to_real_y(y):
    if y in default_digits:
        return int(y)
    return int(int(ord(y) - ord('A')) + 10)


class image_dataset(data.Dataset):
    def __init__(self,
                 root,
                 transform=None,
                 char_set=None,
                 people_set=None,
                 phone_set=None,
                 angle_set=None,
                 env_set=None,
                 hand_set=None):
        self.root = root
        self.transform = transform
        self.char_set = char_set
        self.people_set = people_set
        self.phone_set = phone_set
        self.angle_set = angle_set
        self.env_set = env_set
        self.hand_set = hand_set
        if hand_set is None:
            self.hand_set = default_hand_set
        if env_set is None:
            self.env_set = default_env_set
        if angle_set is None:
            self.angle_set = default_angle_set
        if phone_set is None:
            self.phone_set = default_phone_set
        if people_set is None:
            self.people_set = default_people_set
        if char_set is None:
            self.char_set = default_char_set
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
        for root, dirs, files in os.walk(self.root):
            for file in files:
                file_name = file.split('-')
                class_name = str(os.path.basename(os.path.dirname(os.path.join(root, file))))
                if len(file_name) != 7:
                    continue
                if class_name not in self.char_set:
                    continue
                if file_name[0] not in self.people_set:
                    continue
                if file_name[1] not in self.phone_set:
                    continue
                if file_name[2] not in self.angle_set:
                    continue
                if file_name[4] not in self.env_set:
                    continue
                if file_name[5] not in self.hand_set:
                    continue
                task_key = file_name[0] + file_name[1] + file_name[2] + file_name[4] + file_name[5]
                path = os.path.join(root, file)
                image_data = io.imread(path)
                image_data = skimage.img_as_ubyte(image_data)
                image_data = np.delete(image_data, -1, axis=2)
                img_data.append(image_data)
                cd = to_real_y(class_name)
                if cd not in class_dict.keys():
                    class_dict[cd] = []
                    class_dict[cd].append(i)
                else:
                    class_dict[cd].append(i)
                fn = file_name[0][1:]
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
