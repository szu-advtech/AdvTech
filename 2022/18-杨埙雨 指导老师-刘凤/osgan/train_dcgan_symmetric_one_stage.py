import os
import time
import argparse
import math

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import datasets, transforms
from torchvision.utils import make_grid

from utils.reader import load_yaml, flatten_dict
from utils.logger import Logger
from utils.nn_tools import FeatureHook, GradientHook
from utils.registry import get_model, get_optimizer, get_scheduler

logger = None
cfg = None
num_iter_global = 1


def train_epoch_adv(generator, discriminator, train_loader, optimizers, cfg, hooks=None):
    generator.cuda().train()
    discriminator.cuda().train()

    optimizer_g, optimizer_d = optimizers

    if isinstance(generator, torch.nn.DataParallel):
        zdim = generator.module.zdim
    else:
        zdim = generator.zdim

    global num_iter_global
    loss_adv_value = 0.0

    for i, (x, _) in enumerate(train_loader):
        batch_size = x.size(0)
        x = x.cuda()

        y_real = torch.ones(batch_size).cuda()
        y_fake = torch.zeros(batch_size).cuda()

        z = torch.randn((batch_size, zdim, 1, 1)).cuda()
        fake = generator(z)

        pred_real = discriminator(x)
        pred_fake = discriminator(fake)

        loss_real = F.binary_cross_entropy_with_logits(pred_real, y_real)
        loss_fake = F.binary_cross_entropy_with_logits(pred_fake, y_fake)
        print(loss_fake,loss_real)
        loss_adv = loss_real + loss_fake

        optimizer_g.zero_grad()
        optimizer_d.zero_grad()

        loss_adv.backward()

        loss_val = loss_adv.item()
        loss_real_val = loss_real.item()
        loss_fake_val = loss_fake.item()

        optimizer_g.step()
        optimizer_d.step()

        if logger:
            logger.add_scalar('loss_adv_iter', loss_val, num_iter_global)
            logger.add_scalar('loss_real_iter', loss_real_val, num_iter_global)
            logger.add_scalar('loss_fake_iter', loss_fake_val, num_iter_global)

        loss_adv_value += loss_adv.item() * batch_size

        num_iter_global += 1

    loss_adv_value /= len(train_loader.dataset)
    print(loss_adv_value)
    return loss_adv_value, None


def train(args=None):
    # ---------------- Configure ----------------
    global cfg
    cfg = load_yaml(args.cfg)
    cfg = dict(cfg)
    global logger
    logger = Logger(log_root='log/',
                    name="{}-one-stage_{}-symmetric".format(cfg['method'],
                                                  cfg['dataset']))

    for k, v in flatten_dict(cfg).items():
        logger.add_text('configuration', "{}: {}".format(k, v))

    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg['gpu_id'])

    # ---------------- Dataset ----------------
    if cfg['dataset'] == 'mnist':
        transform = transforms.Compose([
            transforms.Resize(cfg['image_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])
        train_loader = DataLoader(
            datasets.MNIST(cfg['data_root'], train=True, transform=transform),
            cfg['batch_size'], shuffle=True, num_workers=0)
    elif cfg['dataset'] in ['celeba', 'imagenet', 'FFHQ']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        train_loader = DataLoader(datasets.ImageFolder(cfg['data_root'],
                                                       transform=transform),
                                  cfg['batch_size'], shuffle=True, num_workers=0)
        print('Dataset Size:', len(train_loader.dataset))

    # ---------------- Network ----------------
    generator = get_model('{}_generator'.format(cfg['method']), zdim=cfg['zdim'],
                          num_channel=cfg['num_channel'])
    discriminator = get_model('{}_discriminator'.format(cfg['method']),
                              num_channel=cfg['num_channel'])
    generator.weight_init(mean=0.0, std=0.02)
    discriminator.weight_init(mean=0.0, std=0.02)

    # ---------------- Set Hook Func ----------------
    hook_grad = GradientHook(generator)
    hook_grad.set_negate_grads_hook()

    train_epoch = train_epoch_adv
    hooks_bn = None

    if torch.cuda.device_count() > 1:
        generator = torch.nn.DataParallel(generator)
        discriminator = torch.nn.DataParallel(discriminator)

    # ---------------- Optimizer ----------------
    optimizer_g = get_optimizer(cfg['optimizer_g']['type'])( \
        generator.parameters(), **cfg['optimizer_g']['args'])
    optimizer_d = get_optimizer(cfg['optimizer_d']['type'])( \
        discriminator.parameters(), **cfg['optimizer_d']['args'])

    scheduler_g, scheduler_d = None, None
    if 'scheduler_g' in cfg.keys():
        scheduler_g = get_scheduler(cfg['scheduler_g']['type'])(optimizer_g,
                                                                **cfg['scheduler_g']['args'])
    if 'scheduler_d' in cfg.keys():
        scheduler_d = get_scheduler(cfg['scheduler_d']['type'])(optimizer_d,
                                                                **cfg['scheduler_d']['args'])

    # ---------------- Training ----------------
    z_fix = torch.randn((100, cfg['zdim'], 1, 1)).cuda()
    if logger:
        dir_save = 'ckpt/{}'.format(logger.log_name)
    else:
        dir_save = 'ckpt/{}-one-stage_{}'.format(cfg['method'], cfg['dataset'])
    os.makedirs(dir_save, exist_ok=True)

    for epoch in range(1, cfg['num_epoch'] + 1):
        loss_adv, loss_bn = \
            train_epoch(generator, discriminator, train_loader,
                        (optimizer_g, optimizer_d), cfg, (hook_grad, hooks_bn))

        if scheduler_g: scheduler_g.step()
        if scheduler_d: scheduler_d.step()

        if logger:
            logger.add_scalar('loss_adv_epoch', loss_adv, epoch)

            generator.eval()
            fake_fix = generator(z_fix).cpu()
            fake_fix_pack = make_grid(fake_fix, nrow=16, normalize=True,
                                      range=(-1, 1), pad_value=0.5)
            logger.add_image('fake_fix', fake_fix_pack, epoch)

            zs = torch.randn((100, cfg['zdim'], 1, 1)).cuda()
            fake_rand = generator(zs).cpu()
            fake_rand_pack = make_grid(fake_rand, nrow=16, normalize=True,
                                       range=(-1, 1), pad_value=0.5)
            logger.add_image('fake_rand', fake_rand_pack, epoch)

        if isinstance(generator, torch.nn.DataParallel):
            torch.save(generator.module.state_dict(),
                       '{}/generator-epoch{:0>3d}.pth'.format(dir_save, epoch))
        else:
            torch.save(generator.state_dict(),
                       '{}/generator-epoch{:0>3d}.pth'.format(dir_save, epoch))
        if isinstance(discriminator, torch.nn.DataParallel):
            torch.save(discriminator.module.state_dict(),
                       '{}/discriminator-epoch{:0>3d}.pth'.format(dir_save, epoch))
        else:
            torch.save(discriminator.state_dict(),
                       '{}/discriminator-epoch{:0>3d}.pth'.format(dir_save, epoch))


def parse():
    args = argparse.Namespace()
    args.cfg = 'cfgs/dcgan/dcgan_symmetric_one_stage_celeba.yml'
    return args


if __name__ == '__main__':
    args = parse()
    train(args)