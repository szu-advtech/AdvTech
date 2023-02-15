import os
import random
import cv2
import torch
from torch.utils.data import Dataset
from tool.gen_edge_mask import gen_edge_mask
from tool.func import process, read
import numpy as np


class MVSSNetDataset(Dataset):
    def __init__(self, root, txt, val_txt, mode, is_used_edge=False, shuffle=True, seed=None, size=(512, 512)):
        self.size = size
        self.mode = mode
        self.root = root
        self.is_used_edge = is_used_edge
        self.save_list = []
        if self.mode == "val" or self.mode == "test":
            txt = val_txt
        with open(os.path.join(root, txt), "r", encoding='utf-8') as file:
            for line in file:
                line = line.strip('\n')
                line = line.split(' ')
                self.save_list.append(line)
        file.close()
        if shuffle:
            if seed:
                random.seed(seed)
            random.shuffle(self.save_list)
            random.seed(None)
    def __getitem__(self, idx):
        forgery_path = os.path.join(self.root, self.save_list[idx][1])
        mask_path = os.path.join(self.root, self.save_list[idx][2])
        forgery, mask = read(forgery_path, mask_path)
        forgery, mask = process(forgery, mask)

        if self.mode == 'val' or self.mode == 'test':
            self.is_used_edge=False
            return forgery, mask
        if self.is_used_edge:
            label = torch.ones((len(self.save_list),1), dtype=np.float)
            edge_mask = gen_edge_mask(mask)
            edge_mask = cv2.resize(edge_mask, (512 // 4, 512 // 4))
            return forgery, mask, edge_mask, label[idx]
        return forgery, mask

    def __len__(self):
        return len(self.save_list)
        # # 实验三 增加真实图像的训练 casia20.txt
        # elif self.flag == 3:
        #     image_path = os.path.join(self.root, self.save_list[idx][0])
        #     image = cv2.imread(image_path)
        #     image = cv2.resize(image, self.size)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #     image = image.astype(np.float32) / 255.
        #     image = np.transpose(image, (2, 0, 1))
        #     if len(self.save_list[idx]) == 1:
        #         mask = np.zeros(self.size, dtype=np.float)
        #         # mask = torch.as_tensor(mask)
        #         label = 0.
        #     else:
        #         mask_path = os.path.join(self.root, self.save_list[idx][1])
        #         mask = cv2.imread(mask_path)
        #         mask = cv2.resize(mask, self.size)
        #         mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        #         mask = mask.astype(np.float32) / 255.
        #         label = 1.
        #     # print("image path", image_path)
        #
        #
        #     edge_mask = gen_edge_mask(mask)
        #     edge_mask = cv2.resize(edge_mask, (512//4, 512//4))
        #     label = torch.as_tensor(label, dtype=float)
        #     return image, mask, edge_mask, label




