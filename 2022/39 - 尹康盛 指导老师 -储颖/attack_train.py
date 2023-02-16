import cv2
import torchvision
from torch.utils.data import DataLoader
from TOG_Adam import TOG
from attack_dataset import Attack_YoloDataset, yolo_dataset_collate
from utils.utils import get_classes
from yolo import YOLO
from tqdm import tqdm
from PIL import Image

if __name__ == "__main__":
    #------------------------------------------------------------------#
    #   num_workers     用于设置是否使用多线程读取数据
    #                   开启后会加快数据读取速度，但是会占用更多内存
    #                   内存较小的电脑可以设置为2或者0
    #------------------------------------------------------------------#
    num_workers         = 4
    # ----------------------------------------------------#
    #   输入图像size
    # ----------------------------------------------------#
    input_shape = [416, 416]
    #---------------------------------------------------------------------#
    #   classes_path    指向model_data下的txt，与自己训练的数据集相关
    #                   训练前一定要修改classes_path，使其对应自己的数据集
    #---------------------------------------------------------------------#
    classes_path    = 'model_data/voc_classes.txt'
    #----------------------------------------------------#
    #   获取classes和anchor
    #----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    #----------------------------------------------------#
    #   获得图片路径和标签
    #----------------------------------------------------#
    train_annotation_path   = '2007_val.txt'
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    # ---------------------------------------#
    #   构建数据集加载器。
    # ---------------------------------------#
    train_dataset = Attack_YoloDataset(train_lines, input_shape, num_classes, train=False)

    yolo = YOLO()
    train_sampler = None
    shuffle = False
    batch_size = 1
    num = 20
    tog = TOG(yolo.net)

    gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)



    for iteration, batch in enumerate(gen):
        if iteration >= num and num!=0:
            break
        pbar = tqdm(desc=f'picture {iteration}/{len(gen)}', postfix=dict, mininterval=0.3)
        metas, filenames, images, targets = batch[0], batch[1], batch[2], batch[3]
        adv_images = tog(images, targets)
        # for n in range(len(adv_images)):
        #     y1, x1, y2, x2, size = metas[n]
        #     w,h = size
        #     img = adv_images[n, :,x1:x2 , y1:y2]
        #     img = torchvision.transforms.ToPILImage()(img)
        #     img = img.resize((w,h), Image.BICUBIC)
        #     img.save(str(iteration)+".jpg")
        for n in range(len(adv_images)):

             img = adv_images[n, :,: , :]
             img = torchvision.transforms.ToPILImage()(img)
             img.save(str(iteration)+".jpg")




