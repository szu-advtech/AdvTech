import torch
import glob
import numpy as np
import os

from torchvision.io import read_image
from tqdm import tqdm
from utils.color_utils import read_image
from utils.ray_utils import get_ray_directions
from .base import BaseDataset


class NSVFDataset(BaseDataset):
    def __init__(self, root_dir, split='train', **kwargs):
        self.downsample = kwargs.get("downsample")
        self.batch_size = kwargs.get("batch_size")
        super().__init__(root_dir, split, self.downsample)
        self.read_intrinsics()
        if kwargs.get('read_meta', True):
            xyz_min, xyz_max = \
                np.loadtxt(os.path.join(root_dir, 'bbox.txt'))[:6].reshape(2, 3)
            self.shift = (xyz_max + xyz_min) / 2
            self.scale = (xyz_max - xyz_min).max() / 2 * 1.05  # 放大一点
            #  硬代码修复了一些场景的绑定错误
            if 'Mic' in self.root_dir:
                self.scale *= 1.2
            elif 'Lego' in self.root_dir:
                self.scale *= 1.1
            self.read_meta(split)
        self.dataset_name = "nsvf"

    def read_intrinsics(self):
        with open(os.path.join(self.root_dir, 'intrinsics.txt')) as f:
            fx = fy = float(f.readline().split()[0]) * self.downsample
        w = h = int(800 * self.downsample)
        K = np.float32([[fx, 0, w / 2],
                        [0, fy, h / 2],
                        [0, 0, 1]])
        self.K = torch.FloatTensor(K)
        self.directions = get_ray_directions(h, w, self.K)
        self.img_wh = (w, h)

    def read_meta(self, split):
        self.rays = []
        self.poses = []

        if split == 'train':
            prefix = '0_'
        elif split == 'trainval':
            prefix = '[0-1]_'
        elif split == 'val':
            prefix = '1_'
        elif 'Synthetic' in self.root_dir:
            prefix = '2_'  # test set for synthetic scenes
        else:
            raise ValueError(f'{split} split not recognized!')
        img_paths = sorted(glob.glob(os.path.join(self.root_dir, 'rgb', prefix + '*.png')))
        poses = sorted(glob.glob(os.path.join(self.root_dir, 'pose', prefix + '*.txt')))

        print(f'Loading {len(img_paths)} {split} images ...')
        for img_path, pose in tqdm(zip(img_paths, poses)):
            c2w = np.loadtxt(pose)[:3]
            c2w[:, 3] -= self.shift
            c2w[:, 3] /= 2 * self.scale  # to bound the scene inside [-0.5, 0.5] 把场景绑定在里面
            self.poses += [c2w]
            img = read_image(img_path, self.img_wh)
            self.rays += [img]
        self.rays = torch.FloatTensor(np.stack(self.rays))  # (N_images, hw, ?)
        self.poses = torch.FloatTensor(self.poses)  # (N_images, 3, 4)
