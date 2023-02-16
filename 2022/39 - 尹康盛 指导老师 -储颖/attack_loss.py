import torch
import torchvision
from torch.utils.data import DataLoader
from TOG_adam import TOG_adam
from nets.yolo_training import YOLOLoss
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import get_classes, get_anchors
from yolo import YOLO
from tqdm import tqdm
import matplotlib.pyplot as plt


def yololoss():
    anchors_path = 'model_data/yolo_anchors.txt'
    classes_path = 'model_data/voc_classes.txt'
    # ----------------------------------------------------#
    #   获取classes和anchor
    # ----------------------------------------------------#
    class_names, num_classes = get_classes(classes_path)
    anchors, num_anchors = get_anchors(anchors_path)
    # ----------------------------------------------------#
    #   输入图像size
    # ----------------------------------------------------#
    input_shape = [416, 416]
    # ----------------------------------------------------#
    #   Cuda    是否使用Cuda
    #           没有GPU可以设置成False
    # ---------------------------------#
    Cuda = True
    # ---------------------------------------------------------------------#
    #   anchors_mask    用于帮助代码找到对应的先验框，一般不修改。
    # ---------------------------------------------------------------------#
    anchors_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
    # ----------------------#
    #   获得损失函数
    # ----------------------#
    yolo_loss = YOLOLoss(anchors, num_classes, input_shape, Cuda, anchors_mask)

    return yolo_loss

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
    train_annotation_path   = '2007_attack_test.txt'
    #---------------------------#
    #   读取数据集对应的txt
    #---------------------------#
    with open(train_annotation_path) as f:
        train_lines = f.readlines()
    # ---------------------------------------#
    #   构建数据集加载器。
    # ---------------------------------------#
    train_dataset = YoloDataset(train_lines, input_shape, num_classes, train=False)

    yolo = YOLO()
    train_sampler = None
    shuffle = False
    batch_size = 50
    cuda = True
    val_loss = 0
    plt.figure(figsize=(10, 10), dpi=100)
    yolo_loss = yololoss()

    gen_test = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                     drop_last=True, collate_fn=yolo_dataset_collate, sampler=train_sampler)



    for iteration, batch in enumerate(gen_test):
        pbar = tqdm(desc=f'picture {iteration}/{len(gen_test)}', postfix=dict, mininterval=0.3)
        images, targets = batch[0], batch[1]
        with torch.no_grad():
            #----------------------#
            #   前向传播
            #----------------------#
            outputs         = yolo.net(images)
            loss_value_all  = 0
            #----------------------#
            #   计算损失
            #----------------------#
            for l in range(len(outputs)):
                loss_item = yolo_loss(l, outputs[l], targets)
                loss_value_all  += loss_item
        loss_value  = loss_value_all
        plt.scatter(iteration,loss_value.item())
    plt.show()







