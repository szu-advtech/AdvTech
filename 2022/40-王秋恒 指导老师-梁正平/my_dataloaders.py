import copy
import os
import PIL.Image as Image
import torch
import torch.utils.data as data
import numpy as np
from torchvision.transforms import functional as F
import torch.multiprocessing

class Fingerprint(data.Dataset):
    def __init__(self,
                 transforms = None,
                 dataset="train",
                 train_list = None
                 ):
        super().__init__()

        self.mode = dataset
        self.transforms = transforms

        # dataset 只能取train和val两个参数
        assert dataset in ["train", "val"], 'dataset must be in ["train", "val"]'

        self.img_root = "../TrainAndTest/image_crop_v2"
        self.minutae_root = "../TrainAndTest/minutiae_crop_v2"
        self.pore_root = "../TrainAndTest/pore_crop_v2"
        
        every_nums = 270

        self.path_list = []
        if dataset == "train":
            for i in train_list:
                for j in range(1, every_nums+1):
                    self.path_list.append((os.path.join(self.img_root, f"{i+1}", f"{j}.bmp"),
                                           os.path.join(self.pore_root, f"{i+1}", f"{j}.bmp"),
                                           os.path.join(self.minutae_root, f"{i+1}", f"{j}.bmp")))
        if dataset == "val":
            for i in train_list:
                self.path_list.append((("../PoreGroundTruthSampleimage/" + f"{i+1}" + ".bmp"),("../PoreGroundTruthMarked/" + f"{i+1}" + ".txt"),("../minutiae_coord_v2/" + f"{i+1}" + ".txt")))



    def __getitem__(self, idx):
        path = copy.deepcopy(self.path_list[idx])
        if self.mode == "train":
            fingerprint = Image.open(path[0])
            pore = Image.open(path[1])
            minutiae = Image.open(path[2])
            
            pore_and_minutiae = np.stack((pore, minutiae))
            if self.transforms is not None:
                fingerprint, pore_and_minutiae = self.transforms(fingerprint, pore_and_minutiae)

            return fingerprint, pore_and_minutiae
        
        if self.mode == "val":
            torch.multiprocessing.set_sharing_strategy('file_system')

            fingerprint = Image.open(path[0])
            fingerprint = F.to_tensor(fingerprint)

            coords_pore = []
            with open(path[1]) as f:
                for line in f:
                    tmp = line.split()
                    coords_pore.append((int(tmp[0]), int(tmp[1])))
            coords_pore = np.array(coords_pore)
            coords_pore = F.to_tensor(coords_pore)

            coords_minutiae = []
            with open(path[2]) as f:
                for line in f:
                    tmp = line.split()
                    coords_minutiae.append((int(tmp[0]), int(tmp[1])))
            coords_minutiae = np.array(coords_minutiae)
            coords_minutiae = F.to_tensor(coords_minutiae)

            return fingerprint, coords_pore, coords_minutiae

    def __len__(self):
        return len(self.path_list)






