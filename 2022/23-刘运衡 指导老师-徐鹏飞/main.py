# %%
import datetime
import torchvision.transforms as T
# 加载DataLoader
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
from torch import nn
from tqdm import tqdm

from dataset import BaseDataSet
import dataset as D
import os
from os.path import join

from modules import HAModel
from loss import HALoss, advModel
from utils import save_sample, set_folder, log
import numpy as np

USE_DDP = False
USE_DP = True

SAMPLE_ITER = 200
LOG_ITER = 10
EVAL_EPOCH = 1
SAVE_EPOCH = 1
# TODO 用角度表示法向 roughness只用单通道表示 总通道数减少3
LESS_CHANNEL = False
# 计算l1 loss前是否进行对数变换
USE_LOG = True
RESUME = ''
RESULT_DIR = './results'
DATA_DIR = 'D:/data/Data_Deschaintre18/trainBlended'
EVAL_DATA_DIR = 'D:/data/Data_Deschaintre18/testBlended'
# epoch到达一定次数后才开始训练判别器
EPOCHS = 100
SWITCH_EPOCH = 60

LOG_DIR = join(RESULT_DIR, 'log')
SAMPLE_DIR = join(RESULT_DIR, 'sample')
SAVE_DIR = join(RESULT_DIR, 'checkpoints')
set_folder(RESULT_DIR)
set_folder(LOG_DIR)
set_folder(SAMPLE_DIR)
set_folder(SAVE_DIR)

BATCH_SIZE = 6
NUM_WORKERS = 6


def setup():
    dist.init_process_group(backend="nccl")
    torch.manual_seed(42)


def cleanup():
    dist.destroy_process_group()


def train_step(models, loss_func, inputs, label, optimizers, loop, iter=0, epoch=0, train_logger=None):
    """
    models:[model adv1, adv2]
    inputs:
    labels:
    """
    model = models[0]
    [m.train() for m in models]

    output1, output2, output3, output4 = model(inputs)
    # 对抗训练
    if len(models) > 1:
        adversary1 = models[1]
        adversary2 = models[2]
        output1_adv = adversary1(output1, label[:, 0:3, :, :])
        output2_adv = adversary2(output2, label[:, 3:6, :, :])

    output = torch.cat((output1, output2, output3, output4), 1)

    # 计算损失
    map_loss, render_loss, adv_loss = loss_func(
        output, label, [output1_adv, output2_adv] if len(models) > 1 else None)
    loss = 100*map_loss+10*render_loss+1*adv_loss
    # 清空梯度
    [opt.zero_grad() for opt in optimizers]
    # 反向传播
    loss.backward()
    # 更新参数
    [opt.step() for opt in optimizers]

    if iter % LOG_ITER == 0:
        logs = {'epoch': epoch, 'map_loss': map_loss.item(), 'render_loss': render_loss.item(
        ), 'adv_loss': adv_loss.item(), 'loss': loss.item()}
        if train_logger is not None:
            log(logs=logs, iter=iter+epoch*loop.total, writer=train_logger)

        loop.set_description(f'Epoch [{epoch}/{EPOCHS}]')
        loop.set_postfix(**logs)

    if iter % SAMPLE_ITER == 0:
        save_sample(output, sample_dir=SAMPLE_DIR, less_channel=LESS_CHANNEL,
                    save_name='train_%05depoch_%05diter_out.png' % (epoch, iter))
        save_sample(label, sample_dir=SAMPLE_DIR, less_channel=LESS_CHANNEL,
                    save_name='train_%05depoch_%05diter_gt.png' % (epoch, iter))


def eval_step(model, loss_func, inputs, label, loop, iter=0, epoch=0, eval_logger=None):
    model.eval()
    output1, output2, output3, output4 = model(inputs)
    output = torch.cat((output1, output2, output3, output4), 1)
    map_loss, render_loss, adv_loss = loss_func(
        # output, label, [output1_adv, output4_adv])
        output, label)
    loss = 100*map_loss+10*render_loss+1*adv_loss

    if iter % LOG_ITER == 0:
        logs = {'epoch': epoch, 'map_loss': map_loss.item(), 'render_loss': render_loss.item(
        ), 'adv_loss': adv_loss.item(), 'loss': loss.item()}
        if eval_logger is not None:
            log(logs=logs, iter=iter+epoch*loop.total, writer=eval_logger)

        loop.set_description(f'Epoch [{epoch}/{EPOCHS}]')
        loop.set_postfix(**logs)

    if iter % SAMPLE_ITER == 0:
        save_sample(output, sample_dir=SAMPLE_DIR, less_channel=LESS_CHANNEL,
                    save_name='eval_%05depoch_%05diter_out.png' % (epoch, iter))
        save_sample(label, sample_dir=SAMPLE_DIR, less_channel=LESS_CHANNEL,
                    save_name='eval_%05depoch_%05diter_gt.png' % (epoch, iter))


def unpack_data(data, device='cpu'):
    inputs = data[0].to(device)
    label1 = data[1].to(device)
    label2 = data[2].to(device)
    label3 = data[3].to(device)
    label4 = data[4].to(device)
    label = torch.cat((label1, label2, label3, label4), 1)
    return inputs, label


if __name__ == '__main__':
    print('build logger')
    now = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')
    train_logger = SummaryWriter(join(LOG_DIR, 'train_'+now))
    eval_logger = SummaryWriter(join(LOG_DIR, 'eval_'+now))

    print('set device')
    # 检测机器是否有多张显卡
    if USE_DDP and torch.cuda.device_count() > 1:
        print('available devices:', torch.cuda.device_count())
        setup()
        rank = int(os.environ["LOCAL_RANK"])
        device_ids = [rank]
        torch.cuda.set_device(rank)
    else:
        rank = 0
        device_ids = range(torch.cuda.device_count())

    device = torch.device(
        "cuda:"+str(rank) if torch.cuda.is_available() else "cpu")
    print('current device:', device)

    print('build dataset')
    # TODO augment
    transform = D.Compose([
        # D.RandomFlip(p=0.5),
        # D.RandomResize(scale=[1.0, 1.5]),
        # T.CenterCrop(size=[288, 288]),
        T.ToTensor()])
    data_full = BaseDataSet(DATA_DIR, transform, less_channel=LESS_CHANNEL)
    train_size = int(len(data_full)*0.95)
    train_dataset, eval_dataset = random_split(
        data_full, [train_size, len(data_full)-train_size])
    try:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset)
    except:
        pass
    print(len(train_dataset), 'train data')
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        # drop_last=True,
        shuffle=not USE_DDP,
        sampler=train_sampler if USE_DDP else None
    )
    eval_dataset.transform = T.ToTensor()
    try:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(
            eval_dataset)
    except:
        pass
    print(len(eval_dataset), 'eval data loaded')
    eval_loader = DataLoader(
        dataset=eval_dataset,
        batch_size=BATCH_SIZE,
        num_workers=4,
        # drop_last=True,
        shuffle=False,
        sampler=eval_sampler if USE_DDP else None
    )

    print('build model')
    model = HAModel(less_channel=LESS_CHANNEL)
    loss_func = HALoss(renderer=None, less_channel=LESS_CHANNEL,
                       use_log=USE_LOG).to(device)
    # GAN loss for normal and diffuse map
    adversary1 = advModel()
    adversary2 = advModel()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    optimizer_adv1 = torch.optim.Adam(adversary1.parameters(), lr=0.0002)
    optimizer_adv2 = torch.optim.Adam(adversary2.parameters(), lr=0.0002)

    if USE_DDP:
        model = DDP(model.to(device), device_ids=device_ids,
                    output_device=device_ids[0], find_unused_parameters=True)
        adversary1 = DDP(adversary1.to(device), device_ids=device_ids,
                         output_device=device_ids[0], find_unused_parameters=True)
        adversary2 = DDP(adversary2.to(device), device_ids=device_ids,
                         output_device=device_ids[0], find_unused_parameters=True)
    if USE_DP:
        model = nn.DataParallel(model, device_ids=device_ids)
        # adversary1 = nn.DataParallel(adversary1,device_ids=device_ids)
        # adversary2 = nn.DataParallel(adversary2,device_ids=device_ids)
    model = model.to(device)
    adversary1 = adversary1.to(device)
    adversary2 = adversary2.to(device)

    # summary(model=model, input_size=(3,256,256))

    if RESUME != '':
        print('loading from:', RESUME)
        model.load_state_dict(torch.load(RESUME))
        # TODO 加载判别器

    print('start training')
    for epoch in range(EPOCHS):
        total = len(train_loader)
        loop = tqdm(enumerate(train_loader), total=total)
        for i, datas in loop:
            inputs, label = unpack_data(data=datas, device=device)
            train_step([model, adversary1, adversary2] if epoch > SWITCH_EPOCH else [model], loss_func, inputs, label, optimizers=[
                       optimizer, optimizer_adv1, optimizer_adv2], loop=loop, iter=i, epoch=epoch, train_logger=train_logger)

        if epoch % SAVE_EPOCH == 0:
            torch.save(model.state_dict(), join(
                SAVE_DIR, '%05d_model.pth' % epoch))
            torch.save(adversary1.state_dict(), join(
                SAVE_DIR, '%05d_dis_normal.pth' % epoch))
            torch.save(adversary2.state_dict(), join(
                SAVE_DIR, '%05d_dis_diffuse.pth' % epoch))
        if epoch % EVAL_EPOCH == 0:
            eval_total = len(eval_loader)
            eval_loop = tqdm(enumerate(eval_loader), total=eval_total)
            for i, datas in eval_loop:
                inputs, label = unpack_data(data=datas, device=device)
                eval_step(model, loss_func, inputs, label, loop=eval_loop,
                          iter=i, epoch=epoch, eval_logger=eval_logger)
