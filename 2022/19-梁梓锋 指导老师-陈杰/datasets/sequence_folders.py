import torch.utils.data as data
import numpy as np
from imageio import imread
from path import Path
import random


def load_as_float(path):
    return imread(path).astype(np.float32)


class SequenceFolder(data.Dataset):
    """A sequence data loader where the files are arranged in this way:
        root/scene_1/0000000.jpg
        root/scene_1/0000001.jpg
        ..
        root/scene_1/cam.txt
        root/scene_2/0000000.jpg
        .

        transform functions must take in a list a images and a numpy array (usually intrinsics matrix)
    """
    # 加载数据集，图片加内参

    def __init__(self, root, seed=None, train=True, sequence_length=3, transform=None, target_transform=None):
        np.random.seed(seed)
        random.seed(seed)
        self.root = Path(root) # root就是传入的数据集的根路径
        scene_list_path = self.root/'train.txt' if train else self.root/'val.txt'
        self.scenes = [self.root/folder[:-1] for folder in open(scene_list_path)] # 可以看出，通过train.txt/val.txt里面的图片路径获取了一个路径列表
        self.transform = transform
        self.crawl_folders(sequence_length) # 传入序列长度

    def crawl_folders(self, sequence_length):
        # crawl爬：遍历文件夹
        sequence_set = []
        demi_length = (sequence_length-1)//2 # 半长度，有啥用，长度为3时,demi_length=1
        shifts = list(range(-demi_length, demi_length + 1)) # (-1, 0, 1)
        shifts.pop(demi_length) # 弹出下标1的数，也就是弹出了0，所以后续ref_imgs不会把tgt_img加入，故序列长度为3时，ref_imgs只有两张图片
        for scene in self.scenes:
            intrinsics = np.genfromtxt(scene/'cam.txt').astype(np.float32).reshape((3, 3)) # 内参矩阵
            imgs = sorted(scene.files('*.jpg')) # PATH.files，寻找PATH下的文件（尾缀为.jpg的文件），并返回所有这些文件对应的PATH组成的列表
            if len(imgs) < sequence_length:
                continue
            for i in range(demi_length, len(imgs)-demi_length): # demi_length其实就是要来标记上一帧，当前帧（目标帧）和下一帧
                # range取这个范围是因为，前demi_length和后demi_length张图片没有前demi_length帧/后demi_length帧
                sample = {'intrinsics': intrinsics, 'tgt': imgs[i], 'ref_imgs': []} # 构成一个样本{内参，目标帧，参照帧}
                for j in shifts:
                    sample['ref_imgs'].append(imgs[i+j])
                sequence_set.append(sample)#将这个样本添加到sequence_set里面
        random.shuffle(sequence_set)#打乱样本集
        self.samples = sequence_set

    def __getitem__(self, index):
        # 获取数据
        sample = self.samples[index]
        tgt_img = load_as_float(sample['tgt'])
        ref_imgs = [load_as_float(ref_img) for ref_img in sample['ref_imgs']]
        if self.transform is not None:
            imgs, intrinsics = self.transform([tgt_img] + ref_imgs, np.copy(sample['intrinsics']))
            tgt_img = imgs[0]
            ref_imgs = imgs[1:]
        else:
            intrinsics = np.copy(sample['intrinsics'])
        # 返回目标图片（float），参照帧(float)，内参，内参的逆(也就是像素坐标系到相机坐标系的逆变换)
        return tgt_img, ref_imgs, intrinsics, np.linalg.inv(intrinsics)

    def __len__(self):
        return len(self.samples)
