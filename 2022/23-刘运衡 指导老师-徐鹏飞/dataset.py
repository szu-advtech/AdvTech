from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils import normal2uv, uv2normal
import pytorch_lightning as pl
import torchvision.transforms as T
import torchvision.transforms.functional as F

# TODO


class Compose(T.Compose):
    def update(self):
        """
        update classes to generate next random number
        """
        for t in self.transforms:
            try:
                t.update()
            except:
                pass


class RandomRotate(object):
    def __init__(self, rot=[-45, 45]) -> None:
        self.rot = rot
        self.r = 0
        self.update()

    def __call__(self, img, is_normal=False):
        return F.rotate(img, angle=self.r, interpolation=T.InterpolationMode.BILINEAR)

    def update(self):
        self.r = np.random.rand()*(max(self.rot)-min(self.rot))+np.mean(self.rot)


class RandomFlip(object):
    def __init__(self, p=0.5) -> None:
        self.p = p
        self.seed1 = 0
        self.seed2 = 0
        self.update()

    def __call__(self, img, is_normal=False):
        if self.seed1 > 0.5:
            img = F.vflip(img)
        if self.seed2 > 0.5:
            img = F.hflip(img)
        return img

    def update(self):
        self.seed1 = np.random.rand()
        self.seed2 = np.random.rand()


class RandomResize(object):
    def __init__(self, scale=[0.5, 1.5], keep_ratio=False) -> None:
        self.keep_ratio = keep_ratio
        self.scale = scale
        self.seed1 = 0
        self.seed2 = 0
        self.update()

    def __call__(self, img, is_normal=False):
        H, W = img.size
        return F.resize(img, size=[int(H*self.seed1), int(W*self.seed2)])

    def update(self):
        self.seed1 = np.random.rand()*(max(self.scale)-min(self.scale))+np.mean(self.scale)
        if self.keep_ratio:
            self.seed2 = self.seed1
        else:
            self.seed2 = np.random.rand()*(max(self.scale)-min(self.scale))+np.mean(self.scale)


class BaseDataSet(Dataset):
    # 传入训练数据集的路径
    def __init__(self, root_dir, transform=T.ToTensor(), less_channel=False, split=True):
        super().__init__()
        self.root_dir = root_dir
        self.transform = transform
        self.less_channel = less_channel
        tmp = os.listdir(root_dir)
        self.imgs = []
        for f in tmp:
            if any([f.lower().endswith(suffix) for suffix in ['.png', '.bmp', '.jpg']]):
                self.imgs.append(f)
        self.split = split
        self.gray = T.Grayscale(1)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path)

        if not self.split:
            if self.transform:
                image = self.transform(image)
            return image

        # 将图片横向分为五个区域
        image1 = image.crop((0, 0, image.width//5, image.height))
        image2 = image.crop(
            (image.width//5, 0, image.width//5*2, image.height))
        image3 = image.crop(
            (image.width//5*2, 0, image.width//5*3, image.height))
        image4 = image.crop(
            (image.width//5*3, 0, image.width//5*4, image.height))
        image5 = image.crop((image.width//5*4, 0, image.width, image.height))

        # TODO
        self.gray(image4)

        images = [image1, image2, image3, image4, image5]

        if self.transform:
            images = [self.transform(image) for image in images]
            try:
                self.transform.update()
            except:
                pass
        if self.less_channel:
            # TODO 法向转为角度 取值区间[-pi/2,pi/2]转为[0,1]
            images[1] = normal2uv(images[1].unsqueeze(0)*2-1)[0]/torch.pi + 0.5
            # roughness
            images[3] = torch.mean(images[3], dim=0, keepdim=True)
        return images


class DataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset,
                 batch_size: int = 1,
                 num_workers: int = 0,
                 train_rate=0.8,
                 ):
        super().__init__()
        # self.data_dir = data_dir
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_rate = train_rate

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.data_train, self.data_val = self.dataset

        # TODO
        if stage == "test" or stage is None:
            self.data_test = None

    def train_dataloader(self):
        return DataLoader(self.data_train, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.data_val, batch_size=self.batch_size, num_workers=self.num_workers, persistent_workers=True, pin_memory=True)

    def test_dataloader(self):
        return super().test_dataloader()


def test_dataset():
    root_dir = 'D:/data/Data_Deschaintre18/testBlended/'
    transform = Compose([
        # NormalRotate(45),
        # transforms.RandomResizedCrop(size=[288,288]),

        # RandomFlip(p=0.5),
        # RandomResize(scale=[1.0,2.5]),
        # T.CenterCrop(size=[288,288]),
        T.ToTensor()])
    trainset = BaseDataSet(root_dir, transform, less_channel=False)

    print(len(trainset))
    name = ['real', 'normal', 'diffuse', 'roughness', 'specular']
    trainloader = DataLoader(trainset, batch_size=4,
                             shuffle=True, num_workers=0)
    for i, datas in enumerate(trainloader):
        j = 0
        for data in datas:
            img = data[0].numpy()
            img = img.transpose(1, 2, 0)
            img = img * 255
            img = img.astype(np.uint8)
            if img.shape[-1] == 2:
                img = np.concatenate(
                    [np.zeros([288, 288, 1], dtype='uint8'), img], axis=-1)
                plt.imshow(img)
            elif img.shape[-1] == 1:
                img = np.repeat(img, 3, axis=-1)
                plt.imshow(img)
            else:
                plt.imshow(img)
            import cv2
            cv2.imwrite('./split/%05d_%s.png' %
                        (i, name[j]), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            j += 1
