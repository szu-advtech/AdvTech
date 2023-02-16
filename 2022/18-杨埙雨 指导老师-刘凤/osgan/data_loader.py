from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from PIL import Image
import torch
import os
import random
import numpy as np

#borrow some code from StarGAN
class CelebA(data.Dataset):
    """Dataset class for the CelebA dataset."""

    def __init__(self, image_dir, transform, dataset_path):
        """Initialize the CelebA dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.dataset = np.load(dataset_path,allow_pickle=True)
        self.num_images = len(self.dataset)

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        filename = self.dataset[index][0]
        label = self.dataset[index][1]
        #print(self.image_dir, filename[0])
        image = Image.open(os.path.join(self.image_dir, filename[0]))
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


def get_loader(image_dir, dataset_path, crop_size=128, image_size=64, batch_size=128, num_workers=0):
    """Build and return a data loader."""
    transform = []
    transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = CelebA(image_dir, transform, dataset_path)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers,
                                  drop_last=True)
    return data_loader