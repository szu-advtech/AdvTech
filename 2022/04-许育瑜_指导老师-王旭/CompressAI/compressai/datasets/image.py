# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from pathlib import Path

from PIL import Image,ImageFile
from torch.utils.data import Dataset
import numpy as np
import random

TEST_TRANSFORM=1
TRAIN_TRANSFORM=2

from torchvision import transforms
# train_transforms = transforms.Compose(
    # [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
# )

ImageFile.LOAD_TRUNCATED_IMAGES = True         # 有的图片太大，而不能打开


train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]

def train_transforms(img,crop_size):
    # scale = random.choice(train_scale_array)
    # rgb=np.array(img)
    # sw = int(rgb.shape[0] * scale)  
    # sh = int(rgb.shape[1] * scale)
    # size = (sw,sh)  
    # if sw < crop_size[0] or sh < crop_size[1]:
    #     size = (rgb.shape[0],rgb.shape[1])
        
    
    
    return transforms.Compose(
        [transforms.RandomHorizontalFlip(p=0.5),transforms.ToTensor()]
    )(img)
    
def test_transforms(img,crop_size):
    return transforms.Compose( [transforms.CenterCrop(crop_size), transforms.ToTensor()])(img)


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train",size=(224,224),mode='RGB'):
        if split:
            splitdir = Path(root) / split
        else:
            splitdir = Path(root)

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        # self.samples = [ splitdir / (f+'.jpg') for f in test_list]
        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform
        self.size=size
        self.mode=mode

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert(self.mode).resize((1280,1024))
        # if self.transform:
            # return self.transform(img)
        if self.transform == TEST_TRANSFORM:
            return test_transforms(img,self.size)
        if self.transform == TRAIN_TRANSFORM:
            return train_transforms(img,self.size)
        return img

    def __len__(self):
        return len(self.samples)
