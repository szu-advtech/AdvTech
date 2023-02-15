import cv2
import numpy as np
from torchvision import transforms
import os
from PIL import Image
# from .flip import horizontal_flip, vertical_flip
# from torchvision import transforms

def read(forgery_path, mask_path):
    # forgery = cv2.cvtColor(cv2.imread(forgery_path), cv2.COLOR_BGR2RGB)
    # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    forgery = cv2.imread(forgery_path)
    mask = cv2.imread(mask_path)
    return forgery, mask

def process(forgery, mask, augment=True, size=(512, 512)):
    """

    :param forgery: Forgery image
    :param mask:    Binary image. White area represents tampering.
    :param augment: Data augmentation, default is True. Type:{Flip, Rotate, Resize}
    :return:
    """
    forgery, mask = cv2.resize(forgery, size), cv2.resize(mask, size)
    forgery = cv2.cvtColor(forgery, cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # mask = np.expand_dims(mask, axis=-1) # 实验一
    # forgery, mask = np.transpose(forgery, (2, 0, 1)), np.transpose(mask, (2, 0, 1)) # 实验一
    forgery, mask = forgery.astype(np.float32) / 255., mask.astype(np.float32) / 255.
    forgery = np.transpose(forgery, (2, 0, 1))

    #
    # default_data_transforms = {
    #     'train': transforms.Compose([
    #         # transforms.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.2),
    #         # transforms.RandomAffine(degrees=30, translate=(0.2, 0.2), fillcolor=0, scale=(0.9, 1.1), shear=None),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'val': transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    #     'test': transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    #     ]),
    # }
    # forgery = default_data_transforms['test'](Image.fromarray(forgery))

    return forgery, mask

def get_authenticMask(root, authentic):
    authentic_path = os.path.join(root, authentic)
    authentic = cv2.imread(authentic_path)


def modify_txt(root, txt):
    temp = []
    files = open(os.path.join(root, "casia20.txt"), 'w')
    with open(os.path.join(root, txt), "r", encoding='utf-8') as file:
        for line in file:
            line = line.strip('\n')
            line = line.split(' ')
            files.write(line[0] + '\n')
            files.write(line[1]+" "+line[2]+'\n')
        file.close()
        files.close()
    return files.name


if __name__ == '__main__':
    root = '/data/liaogy/CASIA'
    txt = 'casia2.txt'
    modify_txt(root, txt)


