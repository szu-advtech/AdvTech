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
import shutil
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
from torchvision import transforms  # transforms的操作，大多针对 CHW格式，即使是depth，也会变成 1HW
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True         # 有的图片太大，而不能打开

test_list = [
    "FLIR_08884",
    "FLIR_09042",
    "FLIR_09063",
    "FLIR_09175",
    "FLIR_09218",
    "FLIR_09311",
    "FLIR_09451",
    "FLIR_09673",
    "FLIR_09682",
    "FLIR_09705",
    "FLIR_09706",
    "FLIR_09728",
    "FLIR_09751",
    "FLIR_09792",
    "FLIR_09886",
    "FLIR_09896",
    "FLIR_10082",
    "FLIR_10107",
    "FLIR_10171",
    "FLIR_10217",
]

# height,width为bounding box
def random_crop(img, guided, height,width):  
    x = random.randint(0, guided.shape[1] - height)
    y = random.randint(0, guided.shape[2] - width)
    guided = guided[:,x:x+height, y:y+width]
    img = img[:,2*x:2*(x+height), 2*y:2*(y+width)]  # 卷积通过stride会将形状转换成一样的
    return img, guided

train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]

def test_transforms(x,guided,crop_size=(256,256)):
    """
    x,guided are both tensor in [0,1]
    x.size is twice as guided.size
    """
    H,W = crop_size[0],crop_size[1]
    
    # # random resize
    # scale = random.choice(train_scale_array)
    # sh = int(guided.shape[1] * scale) 
    # sw = int(guided.shape[2] * scale)
    # size = (sh,sw)  if sw < W or sh < H else (guided.shape[1],guided.shape[2])
    
    # guided = transforms.Resize(size)(guided)
    # x = transforms.Resize((1280,1024))(x)
    # x = transforms.Resize((size[0]*2,size[1]*2))(x)  

    # # horizontally flip
    # if random.random() > 0.5:
    #     guided = transforms.functional.hflip(guided)
    #     x = transforms.functional.hflip(x)           # it is expected to be in […, H, W] format

    # random crop
    x = transforms.Resize((1024,1280))(x)  # HW  ==》BCHW
    x,guided = random_crop(x,guided,H,W)

    return x,guided



class ImageFolderTest(Dataset):
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

    def __init__(self, root,size=(224,224)):
        splitdir = Path(root) 
        guided_splitdir = Path(root.replace("RGB","thermal_8_bit"))

        if not splitdir.is_dir() or not guided_splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        


        self.samples = [ splitdir / (f+'.jpg') for f in test_list]
        self.guided_samples = [ guided_splitdir / (f+'.jpeg') for f in test_list]



        for x in self.samples:
            shutil.copy(x,root.replace('RGB','RGB20'))
        for x in self.guided_samples:
            shutil.copy(x,root.replace('RGB','thermal_8_bit20')


        # self.samples = [f for f in sorted(splitdir.iterdir()) if f.is_file() ]
        # self.guided_samples = [f for f in sorted(guided_splitdir.iterdir()) if f.is_file()] 
        
        self.size=size

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert('RGB')   # 
        guided = Image.open(self.guided_samples[index])
        # 转化成tensor能否放在最后一步，这样前面的步骤就可以和tf共用？不行，因为也用到了很多transforms的函数。
        # Converts a PIL Image or numpy.ndarray (H x W x C) in the range [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        img = transforms.ToTensor()(img) 
        guided = transforms.ToTensor()(guided)
        return test_transforms(img,guided,self.size)
       

    def __len__(self):
        return len(self.samples)
