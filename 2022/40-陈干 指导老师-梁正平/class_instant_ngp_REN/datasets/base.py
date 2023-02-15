import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange,repeat


class BaseDataset(Dataset):
    def __init__(self, root_dir, split='train', downsample=1.0):
        """
        :param root_dir: 数据路径
        :param split: 训练 or 测试
        :param downsample: 图像下采样
        """
        self.root_dir = root_dir
        self.split = split
        self.downsample = downsample

    def read_intrinsics(self):
        raise NotImplementedError

    def __len__(self):
        if self.split.startswith('train'):
            return 1000
        return len(self.poses)

    def __getitem__(self, idx):
        if self.split.startswith('train'):
            # training pose is retrieved in train.py
            if self.ray_sampling_strategy == 'all_images':  # randomly select images
                img_idxs = np.random.choice(len(self.poses), self.batch_size)
            elif self.ray_sampling_strategy == 'same_image':  # randomly select ONE image
                img_idxs = np.random.choice(len(self.poses), 1)[0]

            if "rffr" == self.dataset_name:
                w, h = self.img_wh[0], self.img_wh[1]
                chose_num = int(self.batch_size / (4 ** 2))
                patch_size = 4
                n_img_patches = w - patch_size + 1 * h - patch_size + 1
                n_patches = n_img_patches * (len(self.poses) - 1)
                start_idx = torch.randint(high=n_patches, size=(chose_num,))
                pix_idxs = self.pix_idxs + rearrange(start_idx, "n->n 1")
                img_idxs = torch.repeat_interleave((start_idx // (n_img_patches) % len(self.poses)), patch_size ** 2)
                pix_idxs = torch.flatten(pix_idxs)

                img_pix_idxs = pix_idxs % (n_img_patches)

                rays = self.rays[img_idxs, pix_idxs]

                masks = self.mask[img_idxs, pix_idxs]
                masks_value = self.all_masks_valid[img_idxs, pix_idxs]
                sample = {'img_idxs': img_idxs, 'pix_idxs': img_pix_idxs, 'rgb': rays[:, :3], "mask": masks,
                          "masks_value": masks_value,
                          "img_wh": self.img_wh, "patch_wh": (patch_size, patch_size)}
            else:
                pix_idxs = np.random.choice(self.img_wh[0] * self.img_wh[1], self.batch_size)
                rays = self.rays[img_idxs, pix_idxs]
                sample = {'img_idxs': img_idxs, 'pix_idxs': pix_idxs, 'rgb': rays[:, :3]}
        else:
            sample = {'pose': self.poses[idx], 'img_idxs': idx}
            if len(self.rays) > 0:  # if ground truth available
                rays = self.rays[idx]
            sample['rgb'] = rays[:, :3]
        return sample


