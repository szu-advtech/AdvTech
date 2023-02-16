import os
import cv2
import glob
import torch
import numpy as np
from PIL import Image
import imgaug.augmenters as iaa
from torchvision import transforms
from torch.utils.data import Dataset
from perlin import rand_perlin_2d_np



class MVTecDataset(Dataset):
    def __init__(self, root, transform, gt_transform, phase):
        # 加载训练集中无异常的图片
        if phase == 'train':
            self.img_path = os.path.join(root, 'train')
        else:
            self.img_path = os.path.join(root, 'test')
            self.gt_path = os.path.join(root, 'ground_truth')
        self.transform = transform
        self.gt_transform = gt_transform
        # load dataset
        '''
        img_paths: 当前阶段所有图片的路径
        gt_paths：如果是train阶段，该参数是无异常图片相等数量的0，如果是test阶段，这个时候是跟img_paths图片中对应的ground_truth图片
        labels：表示图片的标签，train阶段是跟图片有同等数量的0，test阶段就是全部1
        types：图片所属的类型，train阶段是跟图片相同数量的字符串good，测试阶段是跟对应类型图片数量的类型字符串，例如rec
        '''
        self.img_paths, self.gt_paths, self.labels, self.types = self.load_dataset()  # self.labels => good : 0, anomaly : 1

    def load_dataset(self):

        img_tot_paths = []  # 所有的图片路径，例如所有good的训练的图片的路径
        gt_tot_paths = []
        tot_labels = []
        tot_types = []
        # 检测的类型
        defect_types = os.listdir(self.img_path)

        for defect_type in defect_types:
            if defect_type == 'good':
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")  # 查找图片目录下所有的png图片
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend([0] * len(img_paths))  # 图片数量那么多个0
                tot_labels.extend([0] * len(img_paths))
                tot_types.extend(['good'] * len(img_paths))
            else:
                img_paths = glob.glob(os.path.join(self.img_path, defect_type) + "/*.png")
                gt_paths = glob.glob(os.path.join(self.gt_path, defect_type) + "/*.png")
                img_paths.sort()
                gt_paths.sort()
                img_tot_paths.extend(img_paths)
                gt_tot_paths.extend(gt_paths)
                tot_labels.extend([1] * len(img_paths))
                tot_types.extend([defect_type] * len(img_paths))

        assert len(img_tot_paths) == len(gt_tot_paths), "Something wrong with test and ground truth pair!"

        return img_tot_paths, gt_tot_paths, tot_labels, tot_types

    # 定义如下方法应用于Dataloader的batch划分
    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path, gt, label, img_type = self.img_paths[idx], self.gt_paths[idx], self.labels[idx], self.types[idx]
        # 直接open是RGBA是四通道的值
        img = Image.open(img_path).convert('RGB')
        # resize-> tensor -> 裁剪 -> 归一化
        img = self.transform(img)
        # 表示是训练阶段
        if gt == 0:
            # 1通道，然后宽高等于图片的高？
            gt = torch.zeros([1, img.size()[-2], img.size()[-2]])
        else:
            gt = Image.open(gt)
            gt = self.gt_transform(gt)

        assert img.size()[1:] == gt.size()[1:], "image.size != gt.size !!!"
        # os.path.basename(img_path[:-4])拿到图片的文件名并且是不带.png的
        return img, gt, label, os.path.basename(img_path[:-4]), img_type


class MVTecDRAEMTestDataset(Dataset):

    def __init__(self, root_dir, resize_shape=None, transform=None, gt_transform=None):
        self.root_dir = root_dir
        self.images = sorted(glob.glob(root_dir +'/test'+"/*/*.png"))
        self.resize_shape=resize_shape
        self.transform = transform
        self.ge_transform = gt_transform
    def __len__(self):
        return len(self.images)

    def transform_image(self, image_path, mask_path):
        # image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        if mask_path is not None:
            # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = Image.open(mask_path)
            mask = self.ge_transform(mask)
        else:
            # mask = np.zeros((image.shape[1],image.shape[0]))
            mask = torch.zeros([1, image.size()[-2], image.size()[-2]])
        # if self.resize_shape != None:
        #     image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        #     mask = cv2.resize(mask, dsize=(self.resize_shape[1], self.resize_shape[0]))

        # image = image / 255.0
        # mask = mask / 255.0
        # image = np.array(image).reshape((image.shape[0], image.shape[1], 3)).astype(np.float32)
        # mask = np.array(mask).reshape((mask.shape[0], mask.shape[1], 1)).astype(np.float32)
        # image = np.transpose(image, (2, 0, 1))
        # mask = np.transpose(mask, (2, 0, 1))
        return image, mask

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.images[idx]
        dir_path, file_name = os.path.split(img_path)
        base_dir = os.path.basename(dir_path)
        if base_dir == 'good':
            image, mask = self.transform_image(img_path, None)
            has_anomaly = np.array([0], dtype=np.float32)
        else:
            mask_path = os.path.join(dir_path, '../../ground_truth/')
            mask_path = os.path.join(mask_path, base_dir)
            mask_file_name = file_name.split(".")[0]+"_mask.png"
            mask_path = os.path.join(mask_path, mask_file_name)
            image, mask = self.transform_image(img_path, mask_path)
            has_anomaly = np.array([1], dtype=np.float32)
        # sample = {'image': image, 'has_anomaly': has_anomaly,'mask': mask, 'idx': idx}
        l = img_path.split('/')
        type = l[-2]
        file_name = l[-1][:-4]
        return image, mask, has_anomaly, file_name, type, idx

class MVTecDRAEMTrainDataset(Dataset):

    def __init__(self, root_dir, anomaly_source_path, resize_shape=None, transform=None, gt_transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.resize_shape=resize_shape
        self.resize = transforms.Resize((self.resize_shape[1], self.resize_shape[0]), Image.ANTIALIAS)
        self.ToTensor = transforms.ToTensor()
        self.normalize =  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.data_transform = transforms.Compose([
            self.ToTensor,
            self.normalize
        ])
        # MVT
        self.image_paths = sorted(glob.glob(root_dir+"/train/good/"+"/*.png"))
        # 异常源 DTD
        self.anomaly_source_paths = sorted(glob.glob(anomaly_source_path+"/*/*.jpg"))
        # 10种数据增强方法用于训练
        self.augmenters = [iaa.GammaContrast((0.5,2.0),per_channel=True),
                      iaa.MultiplyAndAddToBrightness(mul=(0.8,1.2),add=(-30,30)),
                      iaa.pillike.EnhanceSharpness(),
                      iaa.AddToHueAndSaturation((-50,50),per_channel=True),
                      iaa.Solarize(0.5, threshold=(32,128)),
                      iaa.Posterize(),
                      iaa.Invert(),
                      iaa.pillike.Autocontrast(),
                      iaa.pillike.Equalize(),
                      iaa.Affine(rotate=(-45, 45))
                      ]
        # 旋转
        self.rot = iaa.Sequential([iaa.Affine(rotate=(-90, 90))])


    def __len__(self):
        return len(self.image_paths)

    # 获取3种随机增强方式
    def randAugmenter(self):
        aug_ind = np.random.choice(np.arange(len(self.augmenters)), 3, replace=False)
        aug = iaa.Sequential([self.augmenters[aug_ind[0]],
                              self.augmenters[aug_ind[1]],
                              self.augmenters[aug_ind[2]]]
                             )
        return aug

    def augment_image(self, image, anomaly_source_path):
        # 获取三种随机增强方式
        aug = self.randAugmenter()
        perlin_scale = 6
        min_perlin_scale = 0
        anomaly_source_img = cv2.imread(anomaly_source_path)
        anomaly_source_img = cv2.resize(anomaly_source_img, dsize=(self.resize_shape[1], self.resize_shape[0]))
        # anomaly_source_img = Image.open(anomaly_source_path).convert('RGB')
        # anomaly_source_img = self.resize(anomaly_source_img)
        # 使用随机的3中数据增强方法处理异常源图片
        anomaly_img_augmented = aug(image=anomaly_source_img)
        # 柏林噪声范围
        perlin_scalex = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        perlin_scaley = 2 ** (torch.randint(min_perlin_scale, perlin_scale, (1,)).numpy()[0])
        # 柏林噪声产生，用来模拟乱中有序的现象，如大自然中大理石的纹理、木头的纹理、云的卷动，虽然是随机的，但是并不是完全没有规律的
        # 柏林噪声是一个非常常见的游戏开发技术，主要用于随机地图的生成
        perlin_noise = rand_perlin_2d_np((self.resize_shape[1], self.resize_shape[0]), (perlin_scalex, perlin_scaley))
        perlin_noise = self.rot(image=perlin_noise)
        # 阈值
        threshold = 0.5
        # 根据阈值转成灰度图，这里转出来的就是M_a
        perlin_thr = np.where(perlin_noise > threshold, np.ones_like(perlin_noise), np.zeros_like(perlin_noise))
        # 转成由通道的形式，也就是H*W*C
        perlin_thr = np.expand_dims(perlin_thr, axis=2)
        # 异常部分
        img_thr = anomaly_img_augmented.astype(np.float32) * perlin_thr / 255.0

        beta = torch.rand(1).numpy()[0] * 0.8
        # 产生最终的异常图片
        augmented_image = image * (1 - perlin_thr) + (1 - beta) * img_thr + beta * image * (perlin_thr)

        # no_anomaly = torch.rand(1).numpy()[0]
        # if no_anomaly > 0.5:
        #     image = image.astype(np.float32)
        #     return image, np.zeros_like(perlin_thr, dtype=np.float32), torch.tensor([0.0],dtype=torch.float32)
        # else:
        augmented_image = augmented_image.astype(np.float32)
        msk = (perlin_thr).astype(np.float32)
        augmented_image = msk * augmented_image + (1-msk)*image
        has_anomaly = 1.0
        if np.sum(msk) == 0:
            has_anomaly=0.0
        return augmented_image, msk, torch.tensor([has_anomaly],dtype=torch.float32)

    def transform_image(self, image_path, anomaly_source_path):
        image = cv2.imread(image_path)
        image = cv2.resize(image, dsize=(self.resize_shape[1], self.resize_shape[0]))
        # 随机进行旋转图片增强
        # image = Image.open(image_path).convert('RGB')
        # image = self.resize(image)

        # do_aug_orig = torch.rand(1).numpy()[0] > 0.7
        # if do_aug_orig:
        #     image = self.rot(image=image)

        image = np.array(image).reshape((image.shape[0], image.shape[1], image.shape[2])).astype(np.float32) / 255.0
        # image = self.Totensor(image)
        # 增强的数据可能有异常也可能无异常   has_anomaly类型是np.array  [1] or [0]
        augmented_image, anomaly_mask, has_anomaly = self.augment_image(image, anomaly_source_path)
        # 改成C*H*W
        # augmented_image = np.transpose(augmented_image, (2, 0, 1))
        # image = np.transpose(image, (2, 0, 1))
        # anomaly_mask = np.transpose(anomaly_mask, (2, 0, 1))
        augmented_image = cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        augmented_image = self.data_transform(augmented_image)
        image = self.data_transform(image)
        anomaly_mask = self.ToTensor(anomaly_mask)
        return image, augmented_image, anomaly_mask, has_anomaly

    def __getitem__(self, idx):
        # 随机获取一张无异常MVT图片的索引
        idx = torch.randint(0, len(self.image_paths), (1,)).item()
        # 随机获取一张DTD图片的索引
        anomaly_source_idx = torch.randint(0, len(self.anomaly_source_paths), (1,)).item()
        # 返回的图片可能有异常也可能无异常
        img_path = self.image_paths[idx]
        image, augmented_image, anomaly_mask, has_anomaly = self.transform_image(img_path,
                                                                           self.anomaly_source_paths[anomaly_source_idx])
        # sample = {'image': image, "anomaly_mask": anomaly_mask,
        #           'augmented_image': augmented_image, 'has_anomaly': has_anomaly, 'idx': idx}
        l = img_path.split('/')
        type = l[-2]
        file_name = l[-1][:-4]
        return image, anomaly_mask, augmented_image, has_anomaly, file_name, type, idx
