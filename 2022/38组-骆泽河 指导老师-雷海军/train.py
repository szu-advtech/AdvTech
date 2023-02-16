# 导入库
import os

from EncNet import ENCNet

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")
import os.path as osp
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import wandb

# 用于做可视化
Cam_COLORMAP = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                [0, 64, 128], [0, 32, 128], [0, 16, 128], [0, 64, 64], [0, 64, 32],
                [0, 64, 16], [64, 64, 128], [0, 32, 16], [32, 32, 32], [16, 16, 16], [32, 16, 128],
                [192, 16, 16], [32, 32, 196], [192, 32, 128], [25, 15, 125], [32, 124, 23], [111, 222, 113],
                ]

# 32类
Cam_CLASSES = ['Animal', 'Archway', 'Bicyclist', 'Bridge', 'Building', 'Car', 'CartLuggagePram', 'Child',
               'Column_Pole', 'Fence', 'LaneMkgsDriv', 'LaneMkgsNonDriv', 'Misc_Text', 'MotorcycleScooter',
               'OtherMoving', 'ParkingBlock', 'Pedestrian', 'Road', 'RoadShoulder', 'Sidewalk', 'SignSymbol',
               'Sky', 'SUVPickupTruck', 'TrafficCone', 'TrafficLight', 'Train', 'Tree', 'Truck_Bus', 'Tunnel',
               'VegetationMisc', 'Void', 'Wall']

torch.manual_seed(17)


# 自定义数据集CamVidDataset
class CamVidDataset(torch.utils.data.Dataset):

    def __init__(self, images_dir, masks_dir):
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Normalize(),
            ToTensorV2(),
        ])
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

    def __getitem__(self, i):
        # read data
        image = np.array(Image.open(self.images_fps[i]).convert('RGB'))
        mask = np.array(Image.open(self.masks_fps[i]).convert('RGB'))
        image = self.transform(image=image, mask=mask)

        return image['image'], image['mask'][:, :, 0]

    def __len__(self):
        return len(self.ids)


# 设置数据集路径
DATA_DIR = 'dataset/camvid'  # 根据自己的路径来设置
x_train_dir = os.path.join(DATA_DIR, 'train_images')
y_train_dir = os.path.join(DATA_DIR, 'train_labels')
x_valid_dir = os.path.join(DATA_DIR, 'valid_images')
y_valid_dir = os.path.join(DATA_DIR, 'valid_labels')

train_dataset = CamVidDataset(
    x_train_dir,
    y_train_dir,
)
val_dataset = CamVidDataset(
    x_valid_dir,
    y_valid_dir,
)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True, drop_last=True)

model = ENCNet(num_classes=33).cuda()


# model.load_state_dict(torch.load(r"checkpoints/resnet50-11ad3fa6.pth"),strict=False)

def train_batch_ch13(net, X, y, loss, trainer, devices):
    """Train for a minibatch with mutiple GPUs (defined in Chapter 13).
    Defined in :numref:`sec_image_augmentation`"""
    if isinstance(X, list):
        # Required for BERT fine-tuning (to be covered later)
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)[0]

    # 33是类别数, pred.shape[0]是batch_size的大小
    exist_class = torch.FloatTensor([[1 if c in y[i_batch] else 0 for c in range(33)]
                                     for i_batch in range(pred.shape[0])])

    exist_class = exist_class.cuda()
    pred2 = net(X)[1]

    l = loss(pred, y)
    l1 = nn.functional.mse_loss(pred2, exist_class)

    l = l.sum() + 0.2 * l1
    l.backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = d2l.accuracy(pred, y)
    return train_loss_sum, train_acc_sum


def evaluate_accuracy_gpu(net, data_iter, device=None):
    """Compute the accuracy for a model on a dataset using a GPU.
    Defined in :numref:`sec_lenet`"""
    if isinstance(net, nn.Module):
        net.eval()  # Set the model to evaluation mode
        if not device:
            device = next(iter(net.parameters())).device
    # No. of correct predictions, no. of predictions
    metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # Required for BERT Fine-tuning (to be covered later)
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(net(X)[0], y), d2l.size(y))
    return metric[0] / metric[1]


from d2l import torch as d2l
from tqdm import tqdm
import pandas as pd

# 损失函数选用多分类交叉熵损失函数
lossf = nn.CrossEntropyLoss(ignore_index=255)
# 选用adam优化器来训练
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=-1)

# 训练150轮
epochs_num = 150


def train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, scheduler,
               devices=d2l.try_all_gpus(),wandb=None):
    timer, num_batches = d2l.Timer(), len(train_iter)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0, 1],
                            legend=['train loss', 'train acc', 'test acc'])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])

    loss_list = []
    train_acc_list = []
    test_acc_list = []
    epochs_list = []
    time_list = []
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples,
        # no. of predictions
        metric = d2l.Accumulator(4)
        for i, (features, labels) in enumerate(train_iter):
            timer.start()
            l, acc = train_batch_ch13(
                net, features, labels.long(), loss, trainer, devices)
            

            metric.add(l, acc, labels.shape[0], labels.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        

        animator.add(epoch + 1, (None, None, test_acc))
        scheduler.step()
        print(
            f"epoch {epoch + 1} --- loss {metric[0] / metric[2]:.3f} ---  train acc {metric[1] / metric[3]:.3f} --- test acc {test_acc:.3f} --- cost time {timer.sum()}")

        # ---------保存训练数据---------------
        df = pd.DataFrame()
        loss_list.append(metric[0] / metric[2])
        train_acc_list.append(metric[1] / metric[3])
        test_acc_list.append(test_acc)
        epochs_list.append(epoch + 1)
        time_list.append(timer.sum())

        df['epoch'] = epochs_list
        df['loss'] = loss_list
        df['train_acc'] = train_acc_list
        df['test_acc'] = test_acc_list
        df['time'] = time_list
        df.to_excel("EncNet_camvid1.xlsx")
        # ----------------保存模型-------------------
        if np.mod(epoch + 1, 5) == 0:
            torch.save(model.state_dict(), 'EncNet_{epoch + 1}.pth')

if __name__ == '__main__':
    

    train_ch13(model, train_loader, val_loader, lossf, optimizer, epochs_num, scheduler)