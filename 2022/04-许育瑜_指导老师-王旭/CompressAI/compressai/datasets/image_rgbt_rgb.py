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
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import random
from torchvision import transforms  # transforms的操作，大多针对 CHW格式，即使是depth，也会变成 1HW
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True         # 有的图片太大，而不能打开

# height,width为bounding box
def random_crop(img, guided, height,width):  
    #print('guided.shape',guided.shape,height,width)
    x = random.randint(0, guided.shape[1] - height)
    y = random.randint(0, guided.shape[2] - width)
    guided = guided[:,x:x+height, y:y+width]
    img = img[:,2*x:2*(x+height), 2*y:2*(y+width)]  # 卷积通过stride会将形状转换成一样的
    return img, guided

#train_scale_array = [0.5, 0.75, 1, 1.25, 1.5, 1.75]
train_scale_array = [1,1.2,1.4, 1.6,1.8]

def train_transforms(x,guided,crop_size=(512,640)):
    """
    x,guided are both tensor in [0,1]
    x.size is twice as guided.size
    """
    H,W = crop_size[0],crop_size[1]
    guided = transforms.ToTensor()(guided)
    x = transforms.ToTensor()(x)
    
    # random resize
    scale = random.choice(train_scale_array)
    sh = int(guided.shape[1] * scale) 
    sw = int(guided.shape[2] * scale)
    size = (sh,sw)    
    guided = transforms.Resize(size)(guided)
    x = transforms.Resize((size[0]*2,size[1]*2))(x)  
   
    # random crop
    x,guided = random_crop(x,guided,H,W)

    # horizontally flip
    if random.random() > 0.5:
       guided = transforms.functional.hflip(guided)
       x = transforms.functional.hflip(x)           # it is expected to be in […, H, W] format
    
    return x,guided



class ImageFolderRGB(Dataset):
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

    def __init__(self, root,size=(224,224),channel=3):
        print(root)
        self.root = root
        splitdir = Path(root)  
        if channel == 3: # RGB is master
            guided_splitdir = Path(root.replace("RGB","thermal_8_bit"))
        else:  # thermal is master
            guided_splitdir = Path(root.replace("thermal_8_bit","RGB"))

        if not splitdir.is_dir() or not guided_splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        
        self.samples = [f for f in sorted(splitdir.iterdir()) if f.is_file()]
        self.guided_samples = [f for f in sorted(guided_splitdir.iterdir()) if f.is_file()] 

        self.size=size
        self.channel = channel

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        if self.channel == 3:
            img = Image.open(self.samples[index]).convert('RGB')
            guided = Image.open(self.guided_samples[index])
        else:
            # 两份数据集并不完全对称
            img = Image.open(self.samples[index]) 
            guided = Image.open(self.guided_samples[index]).convert('RGB').resize((1280,1024))

            guided = transforms.ToTensor()(guided)
            img = transforms.ToTensor()(img)
            if random.random() > 0.5:
                guided = transforms.functional.hflip(guided)
                img = transforms.functional.hflip(img)           # it is expected to be in […, H, W] format
            return img,guided

        return train_transforms(img,guided)
       

    def __len__(self):
        # 两份数据集并不完全对称
        if self.channel == 3:
            return len(self.samples)
        return len(self.guided_samples)


class ImageFolderRGB1(Dataset):
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

    def __init__(self, root,size=(224,224),channel=3):
        print(root)
        self.root = root
        splitdir = Path(root)  
        if channel == 3: # RGB is master
            guided_splitdir = Path(root.replace("RGB","thermal_8_bit"))
        else:  # thermal is master
            guided_splitdir = Path(root.replace("thermal_8_bit","RGB"))

        if not splitdir.is_dir() or not guided_splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')
        
        self.samples = [f for f in sorted(splitdir.iterdir()) if f.is_file()]
        self.guided_samples = [f for f in sorted(guided_splitdir.iterdir()) if f.is_file()] 

        self.size=size
        self.channel = channel

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        try:
            
            img = Image.open(self.samples[index]).convert('RGB')
            guided = Image.open(self.guided_samples[index])
            
        except Exception as e:
            # 输出到控制台
            traceback.print_exc()
            print('Exception happens at',time.asctime( time.localtime(time.time()) ))
            
            # 输出到文件中
            trace_file_path = 'dataset_error.log'
            trace_file=open(trace_file_path, 'a')
            print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),file=trace_file) #记录报错的时间
            print(f'path:{self.samples[index]}')
            traceback.print_exc(file=trace_file) #输出报错到文件 
            trace_file.close()
            return torch.tensor(1),torch.tensor(2)
        else:
            print('Task is finished.')    
       
        guided = transforms.ToTensor()(guided)
        img = transforms.ToTensor()(img)    

        return img,guided
       

    def __len__(self):
        # 两份数据集并不完全对称
        if self.channel == 3:
            return len(self.samples)
        return len(self.guided_samples)


if __name__ == '__main__':
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    import time 
    import torch
    import traceback
    from torch.utils.data import DataLoader
    

    # train_dataset = ImageFolderRGB1('/data/xyy/FLIR_ADAS_1_3/train/RGB',channel = 3)

    # train_dataloader = DataLoader(
    #     train_dataset,
    #     batch_size=1,
    #     num_workers=0,
    #     shuffle=False,
    #     pin_memory=False,
    # )


    # for i, (x,guided) in enumerate(train_dataloader):
    #     print(x.shape,guided.shape)

    img = Image.open('/data/xyy/FLIR_ADAS_1_3/train/RGB/FLIR_08520.jpg')
    print(img.size)
    arr = np.array(img)
    print(arr.shape)