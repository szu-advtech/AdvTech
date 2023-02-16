import os
import logging
import cv2
import numpy as np
import torch
import requests
import pandas as pd
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def download_file(url, dest):
    CHUNK_SIZE = 8192

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # Filter out keep-alive new chunks.
                f.write(chunk)


def download_file_from_google_drive(id, dest):
    URL = "https://docs.google.com/uc?export=download"

    # 实例化一个对象，达到状态保持#
    session = requests.Session()

    # 发出get请求#
    response = session.get(URL, params={"id": id}, stream=True, verify=False)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True, verify=False)

    save_response_content(response, dest)


class NumpyDataset(Dataset):
    """ Dataset for numpy arrays with explicit memmap support """

    def __init__(self, *arrays, **kwargs):

        self.dtype = kwargs.get("dtype", torch.float)
        """
        作用：内存映像文件是一种将磁盘上的非常大的二进制数据文件当做内存中的数组进行处理的方式。NumPy实现了一个类似于ndarray的memmap对象，它允许将大文件分成小段进行读写，而不是一次性将整个数组读入内存。
        memmap也拥有跟普通数组一样的方法，因此，基本上只要是能用于ndarray的算法就也能用于memmap。
        用法：
        1-创建：fp = np.memmap(filename, dtype=‘float32’, mode=‘w+’, shape=(3,4))
        2-赋值：fp[:] = data[:]
        3-删除：del fp
        4-读取：fpr = np.memmap(filename, dtype=‘float32’, mode=‘r’, shape=(3,4))
        """
        self.memmap = []
        self.data = []
        self.n = None

        memmap_threshold = kwargs.get("memmap_threshold", None)

        for array in arrays:
            if isinstance(array, str):
                array = self._load_array_from_file(array, memmap_threshold)

            if self.n is None:
                self.n = array.shape[0]
            assert array.shape[0] == self.n

            if isinstance(array, np.memmap):
                self.memmap.append(True)
                self.data.append(array)
            else:
                self.memmap.append(False)
                tensor = torch.from_numpy(array).to(self.dtype)
                self.data.append(tensor)

    def __getitem__(self, index):
        items = []
        for memmap, array in zip(self.memmap, self.data):
            if memmap:
                tensor = np.array(array[index])
                items.append(torch.from_numpy(tensor).to(self.dtype))
            else:
                items.append(array[index])
        return tuple(items)

    def __len__(self):
        return self.n

    @staticmethod
    def _load_array_from_file(filename, memmap_threshold_gb=None):
        filesize_gb = os.stat(filename).st_size / 1.0 * 1024 ** 3
        if memmap_threshold_gb is None or filesize_gb <= memmap_threshold_gb:
            data = np.load(filename)
        else:
            data = np.load(filename, mmap_mode="c")

        if len(data.shape) == 1:
            data = data.reshape(-1, 1)

        return data


class LabelledImageDataset(Dataset):
    def __init__(self, x, y, transform=None):
        self.transform = transform
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]

        if self.transform is not None:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return self.x.shape[0]


class UnlabelledImageDataset(Dataset):
    def __init__(self, array, transform=None):
        self.transform = transform
        self.data = torch.from_numpy(array)

    def __getitem__(self, index):
        # 获取灰度图, 并转成RGB图
        img = self.data[index, ...].unsqueeze(0)
        img=img.expand(3,img.size(1),img.size(2))
        img=transforms.Resize((64,64))(img)

        if self.transform is not None:
            img = self.transform(img)

        # 判断是否有nan值
        if torch.isnan(img).any():
            logger.info("Ending training since index %d data has NaNs ", index)

        return img, torch.tensor([0.0])

    def __len__(self):
        return self.data.shape[0]


class CSVLabelledImageDataset(Dataset):
    """ Based on https://pytorch.org/tutorials/beginner/data_loading_tutorial.html """

    def __init__(self, csv_file, root_dir, label_key, filename_key, image_transform=None, label_transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.label_key = label_key
        self.filename_key = filename_key

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_filename = os.path.join(self.root_dir, self.df[self.filename_key].iloc[idx])
        image = torch.from_numpy(np.transpose(plt.imread(img_filename), [2, 0, 1]))
        label = torch.tensor([self.df[self.label_key].iloc[idx]], dtype=np.float)

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        return image, label


class Preprocess:
    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.num_bins = 2 ** self.num_bits

    def __call__(self, img):
        """可对取值为[0,255]或者[0,1]的图片做预处理：如果num_bits==8则归一化[0,255]，且加均匀噪声到图片上"""
        if img.dtype == torch.uint8:
            img = img.float()  # Already in [0,255]
        elif torch.max(img) <= 1.0:
            img = img * 255.0  # [0,1] -> [0,255]

        if self.num_bits != 8:
            img = torch.floor(img / 2 ** (8 - self.num_bits))  # [0, 255] -> [0, num_bins - 1]

        # Uniform dequantization.
        img = img + torch.rand_like(img)

        return img

    def inverse(self, inputs):
        # Discretize the pixel values.
        inputs = torch.floor(inputs)
        # Convert to a float in [0, 1].
        inputs = inputs * (256 / self.num_bins) / 255
        inputs = torch.clamp(inputs, 0, 1)
        return inputs


class RandomHorizontalFlipTensor(object):
    """Random horizontal flip of a CHW image tensor."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        assert img.dim() == 3
        if np.random.rand() < self.p:
            return img.flip(2)  # Flip the width dimension, assuming img shape is CHW.
        return img

    def __repr__(self):
        return self.__class__.__name__ + "(p={})".format(self.p)
