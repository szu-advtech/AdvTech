# -*- coding: utf-8 -*-
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class WholeDataLoader(Dataset):

    def __init__(self, option):
        self.data_split = option.data_split
        data = np.load(f'./data/mnist/mnist_var_{option.color_var}.npy', encoding='latin1', allow_pickle=True).item()
        if self.data_split == 'train':
            self.image = data['train_image']
            self.label = data['train_label']
        if self.data_split == 'test':
            self.image = data['test_image']
            self.label = data['test_label']

        self.std = option.color_var ** 0.5

        self.T = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        self.ToPIL = transforms.Compose([
            transforms.ToPILImage(),
        ])

    def __getitem__(self, index):
        image = self.image[index]
        label = self.label[index]
        image = self.ToPIL(image)

        label_image = image.resize((14,14), Image.NEAREST)


        label_image = torch.from_numpy(np.transpose(label_image,(2,0,1)))
        mask_image = torch.lt(label_image.float()-0.00001, 0.) * 255
        label_image = torch.div(label_image,32)
        label_image = label_image + mask_image
        label_image = label_image.long()

        return self.T(image), label_image,  label.astype(np.long)

    def __len__(self):
        return self.image.shape[0]

