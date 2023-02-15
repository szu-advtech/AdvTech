import datetime
from torchvision import transforms
# 加载DataLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.elastic.multiprocessing.errors import record
import cv2
from tqdm import tqdm

from dataset import BaseDataSet
import os
from os.path import join

from modules import HAModel
from loss import HALoss, advModel
from utils import save_sample, set_folder, log, unpack_svbrdf, uv2normal
import numpy as np


@record
def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 计算l1 loss前是否进行对数变换
    USE_LOG = True

    USE_MULTI_GPU = False
    RESUME = './results/adv/checkpoints/00012_model.pth'
    LESS_CHANNEL = False
    REAL_IMG = True

    RESULT_DIR = './results'
    SAVE_GT = True
    DATA_DIR = 'D:/data/BRDF_test/png_1024'

    # LOG_DIR = join(RESULT_DIR, 'log')
    SAMPLE_DIR = join(RESULT_DIR, 'test')
    # SAVE_DIR = join(RESULT_DIR, 'checkpoints')
    set_folder(RESULT_DIR)
    # set_folder(LOG_DIR)
    set_folder(SAMPLE_DIR)
    # set_folder(SAVE_DIR)

    BATCH_SIZE = 1
    NUM_WORKERS = 0

    # log
    # print('build logger')
    # now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    # test_logger = SummaryWriter(join(LOG_DIR,'train_'+now))

    # dataset
    print('build dataset')
    transform = transforms.Compose([
        transforms.ToTensor()])
    test_dataset = BaseDataSet(
        DATA_DIR, transform, less_channel=LESS_CHANNEL, split=not REAL_IMG)

    print(len(test_dataset), 'data loaded')
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        # drop_last=True,
        # shuffle=True,
        shuffle=False,
    )

    print('build model')
    model = HAModel(less_channel=LESS_CHANNEL)
    model = nn.DataParallel(model, device_ids=[0, 1]).to(device)
    # model = nn.DataParallel(model,device_ids=[0]).to(device)

    if RESUME != '':
        print('loading from:', RESUME)
        model.load_state_dict(torch.load(RESUME, map_location=device))

    loss_func = HALoss(renderer=None, less_channel=LESS_CHANNEL,
                       use_log=USE_LOG).to(device)

    # summary(model=model, input_size=(3,256,256))
    total_losses = []
    rmse = []
    print('start testing')
    # 训练模型
    total = len(test_loader)
    loop = tqdm(enumerate(test_loader), total=total)
    for i, datas in loop:
        if REAL_IMG:
            inputs = datas.to(device)
            B, C, H, W = inputs.shape
            label = torch.zeros(
                [B, 9 if LESS_CHANNEL else 12, H, W], device=device)
        else:
            inputs = datas[0].to(device)
            label1 = datas[1].to(device)
            label2 = datas[2].to(device)
            label3 = datas[3].to(device)
            label4 = datas[4].to(device)
            label = torch.cat((label1, label2, label3, label4), 1)

        output1, output2, output3, output4 = model(inputs)
        output = torch.cat((output1, output2, output3, output4), 1)

        map_loss, render_loss, adv_loss = loss_func(
            output, label)

        loss = 100*map_loss+10*render_loss+1*adv_loss

        logs = {'map_loss': map_loss.item(), 'render_loss': render_loss.item(
        ), 'adv_loss': adv_loss.item(), 'loss': loss.item()}
        total_losses.append(
            [map_loss.item(), render_loss.item(), adv_loss.item(), loss.item()])

        rmse.append([
            nn.functional.mse_loss(output, label).item(),
        ])
        loop.set_postfix(**logs)

        save_sample(output, sample_dir=SAMPLE_DIR, less_channel=LESS_CHANNEL, input_img=inputs.detach().cpu(),
                    save_name='test_%05d_out.png' % i)
        if not REAL_IMG and SAVE_GT:
            save_sample(label, sample_dir=SAMPLE_DIR, less_channel=LESS_CHANNEL, input_img=inputs.detach().cpu(),
                        save_name='test_%05d_gt.png' % i)

    print(np.array(total_losses).mean(axis=0))
    print(np.array(rmse).mean(axis=0))


if __name__ == '__main__':
    test()
