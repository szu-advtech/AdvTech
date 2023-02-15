from PIL import Image

from torchvision import transforms


def image_transforms(load_size):

    return transforms.Compose([
        #transforms.CenterCrop(size=(178, 178)),  # for CelebA
        transforms.Resize(size=load_size, interpolation=Image.BILINEAR),   
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def mask_transforms(load_size):

    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 彩色图像转灰度图像num_output_channels默认1
        transforms.Resize(size=load_size, interpolation=Image.NEAREST),
        transforms.ToTensor()
    ])
