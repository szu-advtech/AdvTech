import os
from PIL import Image
import numpy as np
import jittor as jt
from jittor.dataset import Dataset
import jittor.transform as transforms
import albumentations as A
import cv2

VOC_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128]]

VOC_CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'potted plant', 'sheep', 'sofa', 'train', 'tv/monitor']

colormap2label = jt.zeros(256 ** 3, dtype="uint8")
for i, colormap in enumerate(VOC_COLORMAP):
    colormap2label[(colormap[0] * 256 + colormap[1]) * 256 + colormap[2]] = i


def voc_label_indices(colormap):
    """
    convert colormap (PIL image) to colormap2label (uint8 tensor).
    """
    colormap = np.array(colormap).astype('int32')
    idx = ((colormap[:, :, 0] * 256 + colormap[:, :, 1]) * 256
           + colormap[:, :, 2])
    return colormap2label[idx]


def read_file_list(root, is_train=True):
    txt_fname = root + '/ImageSets/Segmentation/' + ('train.txt' if is_train else 'val.txt')
    with open(txt_fname, 'r') as f:
        filenames = f.read().split()
    images = [os.path.join(root, 'JPEGImages', i + '.jpg') for i in filenames]
    labels = [os.path.join(root, 'SegmentationClass', i + '.png') for i in filenames]
    return images, labels  # file list


def voc_rand_crop(image, label, height, width):
    """
    Random crop image (PIL image) and label (PIL image).
    """
    top = np.random.randint(0, height - height + 1)
    left = np.random.randint(0, width - width + 1)
    image = transforms.crop(img=image, left=left, top=top, height=height, width=width)
    label = transforms.crop(img=label, left=left, top=top, height=height, width=width)
    return image, label


class VOCSegDataset(jt.dataset.Dataset):
    def __init__(self, is_train, crop_size, voc_root):
        """
        crop_size: (h, w)
        """
        super().__init__()
        self.is_train = is_train
        self.rgb_mean = np.array([0.485, 0.456, 0.406])
        self.rgb_std = np.array([0.229, 0.224, 0.225])

        if is_train:
            self.rand_transform = A.Compose([
                A.RandomCrop(width=crop_size[1], height=crop_size[0]),
                A.HorizontalFlip(p=0.5),
                A.RandomRotate90(p=0.4),
                A.RandomBrightnessContrast(p=0.2),
                A.Blur(p=0.3),
                A.ToGray(p=0.1),
                A.GaussNoise(p=0.7),
                A.CLAHE(p=0.3),
                A.JpegCompression(p=0.2),
                A.RandomShadow(p=1.)
            ])
            self.transform = A.Compose([

            ])

        self.tsf = transforms.Compose([
            transforms.ToTensor(),
            transforms.ImageNormalize(mean=self.rgb_mean, std=self.rgb_std)
        ])

        self.crop_size = crop_size  # (h, w)
        images, labels = read_file_list(root=voc_root, is_train=is_train)
        self.images = self.filter(images)  # images list
        self.labels = self.filter(labels)  # labels list
        print('Read ' + str(len(self.images)) + ' valid examples')

    def filter(self, imgs):  # 过滤掉尺寸小于crop_size的图片
        return [img for img in imgs if (
                Image.open(img).size[1] >= self.crop_size[0] and
                Image.open(img).size[0] >= self.crop_size[1])]

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.is_train:
            image = cv2.imread(image)
            label = cv2.imread(label)
            # cv2.imwrite('./raw_img.jpg',image)
            # cv2.imwrite('./raw_lb.jpg',label)

            # 非形变，image参与即可
            # transformed = self.transform(image=image)
            # image = transformed['image']
            # cv2.imwrite(' ./1_img.jpg', image)
            # 形变，一起参与
            rand_transformed = self.rand_transform(image=image, mask=label)  # 形参必须为mask
            image, label = rand_transformed['image'], rand_transformed['mask']
            # cv2.imwrite('./2_img.jpg',image)
            # cv2.imwrite('./2_lb.jpg', label)

            # label = Image.fromarray(transformed['mask'])

            image, label = Image.fromarray(image).convert('RGB'), Image.fromarray(label).convert('RGB')
        else:
            image = Image.open(image).convert('RGB')
            label = Image.open(label).convert('RGB')
            image, label = voc_rand_crop(image, label,
                                         *self.crop_size)
        image = self.tsf(image)
        label = voc_label_indices(label)
        return image, label  # float32 tensor, uint8 tensor

    def __len__(self):
        return len(self.images)

# 修改过的出问题版本
# class VOCSegDataset(jt.dataset.Dataset):
#     def __init__(self, is_train, crop_size, voc_root):
#         """
#         crop_size: (h, w)
#         """
#         super().__init__()
#         self.is_train = is_train
#         self.rgb_mean = np.array([0.485, 0.456, 0.406])
#         self.rgb_std = np.array([0.229, 0.224, 0.225])
#         if is_train:
#             self.rand_transform = A.Compose([
#                 A.RandomCrop(width=crop_size[1], height=crop_size[0]),
#                 A.HorizontalFlip(p=0.5),
#                 A.RandomRotate90()
#             ])
#             self.transform = A.Compose([
#                 A.RandomBrightnessContrast(p=0.2),
#                 A.RandomSunFlare(p=0.2),
#                 A.RandomRain(p=0.2),
#                 A.RandomSnow(p=0.2),
#                 A.RandomFog(p=0.2),
#                 A.RandomShadow(p=0.2)
#             ])
#
#         self.tsf = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.ImageNormalize(mean=self.rgb_mean, std=self.rgb_std)
#         ])
#
#         self.crop_size = crop_size  # (h, w)
#         images, labels = read_file_list(root=voc_root, is_train=is_train)
#         self.images = self.filter(images)  # images list
#         self.labels = self.filter(labels)  # labels list
#         print('Read ' + str(len(self.images)) + ' valid examples')
#
#     def filter(self, imgs):  # 过滤掉尺寸小于crop_size的图片
#         return [img for img in imgs if (
#                 Image.open(img).size[1] >= self.crop_size[0] and
#                 Image.open(img).size[0] >= self.crop_size[1])]
#
#     def __getitem__(self, idx):
#         image = self.images[idx]
#         label = self.labels[idx]
#         # image = Image.open(image).convert('RGB')    # H,W
#         # label = Image.open(label).convert('RGB')
#         if self.is_train:
#             image = cv2.imread(image)
#             label = cv2.imread(label)
#             # 形变，一起参与
#             rand_transformed = self.rand_transform(image=image,mask=label)  # 形参必须为mask
#             image, label = rand_transformed['image'],rand_transformed['mask']
#             # 非形变，image参与即可
#             transformed = self.transform(image=image)   # 形参必须为mask
#             # label = Image.fromarray(transformed['mask'])
#             image,label = Image.fromarray(transformed['image']).convert('RGB'),Image.fromarray(label).convert('RGB')
#         else:
#             image = Image.open(image).convert('RGB')    # H,W
#             label = Image.open(label).convert('RGB')
#             image, label = voc_rand_crop(image, label,
#                                          *self.crop_size)
#         image = self.tsf(image)
#         label = voc_label_indices(label)
#
#         # p = Augmentor.Pipeline(image)
#         # p.ground_truth(label)
#         # p.random_brightness(probability=random.random(),min_factor=0.1,max_factor=10)
#         # p.random_contrast(probability=random.random(),min_factor=0.1,max_factor=10)
#         # p.rotate(probability=random.random(),max_left_rotation=90,max_right_rotation=90)
#         # p.flip_left_right(0.3)
#         # p.flip_top_bottom(0.3)
#         # p.sample()
#         return image, label  # float32 tensor, uint8 tensor
#
#     def __len__(self):
#         return len(self.images)
