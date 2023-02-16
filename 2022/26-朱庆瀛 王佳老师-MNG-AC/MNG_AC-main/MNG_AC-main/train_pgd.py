import argparse
import copy
import logging
import math
import random
import sys
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
#import apex.amp as amp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from preact_resnet import resnet50 as ResNet50
from wideresnet import WideResNet
from evaluate import clamp, norms, norms_l1, norms_p
from evaluate import l1_dir_topk, proj_l1ball, proj_simplex
from torch.distributions import laplace

from torch_backend import *
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler, RandomSampler
from collections import OrderedDict
import torch.nn.functional as F
from torch import autograd
from datasets import SemiSupervisedDataset, SemiSupervisedSampler, DATASETS
from datasets import setup_data_loader
from train_MNG import fix_perturbation_size

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG)

cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def initialize_weights(module):
  if isinstance(module, nn.Conv2d):
    n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
    module.weight.data.normal_(0, math.sqrt(2. / n))
    if module.bias is not None:
      module.bias.data.zero_()
  elif isinstance(module, nn.BatchNorm2d):
    module.weight.data.fill_(1)
    module.bias.data.zero_()
  elif isinstance(module, nn.Linear):
    module.bias.data.zero_()


def attack_pgd(model, X, y, opt, norm, dataset, params=None):
  delta = torch.zeros_like(X).cuda()
  if norm == "linf":
    if dataset == "cifar10" or dataset == "svhn":
      epsilon = (8 / 255.) / std
    else:
      epsilon = (4 / 255.) / std
    attack_iters = 10
    alpha = (1 / 255.) / std
    delta[:, 0, :, :].uniform_(-epsilon[0][0][0].item(),
                               epsilon[0][0][0].item())
    delta[:, 1, :, :].uniform_(-epsilon[1][0][0].item(),
                               epsilon[1][0][0].item())
    delta[:, 2, :, :].uniform_(-epsilon[2][0][0].item(),
                               epsilon[2][0][0].item())
  elif norm == "l2":
    if dataset == "cifar10" or dataset == "svhn":
      epsilon = (128 / 255.) / std
    else:
      epsilon = (80 / 255.) / std
    attack_iters = 10
    alpha = (30. / 255.) / std
    delta = torch.rand_like(X, requires_grad=True)
    delta.data *= (2.0 * delta.data - 1.0) * epsilon
    delta.data /= norms_p(
        delta.detach(), 2.0).clamp(min=epsilon.detach().cpu().numpy()[0][0][0])
  elif norm == "l1":
    epsilon = (2000 / 255.) / std
    attack_iters = 20
    alpha = (255. / 255.) / std
    ini = laplace.Laplace(loc=delta.new_tensor(0), scale=delta.new_tensor(1))
    delta.data = ini.sample(delta.data.shape)
    delta.data = (2.0 * delta.data - 1.0) * epsilon
    delta.data /= norms_l1(
        delta.detach()).clamp(min=epsilon.detach().cpu().numpy()[0][0][0])
  delta.requires_grad = True
  for _ in range(attack_iters):
    output = model(X + delta)
    loss = F.cross_entropy(output, y)
    loss.backward()
    grad = delta.grad.detach()
    if norm == "linf":
      delta.data = clamp(delta.data + alpha * torch.sign(grad), -epsilon,
                         epsilon)
    elif norm == "l2":
      delta.data = delta.data + alpha * grad / norms_p(grad, 2.0)
      delta.data *= epsilon / norms_p(delta.detach(), 2.0).clamp(
          min=epsilon.detach().cpu().numpy()[0][0][0])
    elif norm == "l1":
      k = 99
      delta.data = delta.data + alpha * l1_dir_topk(grad, delta.data, X, k)
      delta.data = proj_l1ball(delta.data,
                               epsilon=epsilon.detach().cpu().numpy()[0][0][0],
                               device=device)
    delta.data = clamp(delta.data, lower_limit - X, upper_limit - X)
    delta.grad.zero_()
  return delta.detach()


def attack_msd(model,
               X,
               y,
               opt,
               dataset,
               epsilon_l_inf=8. / 255,
               epsilon_l_2=128. / 255,
               epsilon_l_1=2000. / 255,
               alpha_l_inf=1. / 255,
               alpha_l_2=25. / 255,
               alpha_l_1=255. / 255,
               num_iter=20,
               device="cuda:0"):
  delta = torch.zeros_like(X, requires_grad=True)
  max_delta = torch.zeros_like(X)
  max_max_delta = torch.zeros_like(X)
  max_loss = torch.zeros(y.shape[0]).to(y.device).float()
  max_max_loss = torch.zeros(y.shape[0]).to(y.device).float()
  alpha_l_1_default = alpha_l_1

  for t in range(num_iter):
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    with torch.no_grad():
      #For L_2
      delta_l_2 = delta.data + alpha_l_2 * delta.grad / norms(delta.grad)
      delta_l_2 *= epsilon_l_2 / norms_p(delta_l_2.detach(),
                                         2.0).clamp(min=epsilon_l_2)

      delta_l_2.data = clamp(delta_l_2.data, lower_limit - X, upper_limit - X)

      #For L_inf
      delta_l_inf = (delta.data + alpha_l_inf * delta.grad.sign()).clamp(
          -epsilon_l_inf, epsilon_l_inf)
      delta_l_inf.data = clamp(delta_l_inf.data, lower_limit - X,
                               upper_limit - X)

      #For L1
      k = 99
      delta_l_1 = delta.data + alpha_l_1 * l1_dir_topk(delta.grad, delta.data,
                                                       X, k)
      delta_l_1 = proj_l1ball(delta_l_1, epsilon_l_1, device)
      delta_l_1.data = clamp(delta_l_1.data, lower_limit - X, upper_limit - X)

      #Compare
      delta_tup = (delta_l_1, delta_l_2, delta_l_inf)
      max_loss = torch.zeros(y.shape[0]).to(y.device).float()
      for delta_temp in delta_tup:
        loss_temp = nn.CrossEntropyLoss(reduction='none')(model(X +
                                                                delta_temp), y)
        max_delta[loss_temp >= max_loss] = delta_temp[loss_temp >= max_loss]
        max_loss = torch.max(max_loss, loss_temp)
      delta.data = max_delta.data
      max_max_delta[max_loss > max_max_loss] = max_delta[
          max_loss > max_max_loss]
      max_max_loss[max_loss > max_max_loss] = max_loss[max_loss > max_max_loss]
    delta.grad.zero_()

  return max_max_delta.detach()


def get_loaders(dir_, batch_size, dataset):
  if dataset == "cifar10":
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cifar10_mean, cifar10_std)
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])
  elif dataset == "svhn":
    train_transform = transforms.Compose([transforms.ToTensor()])
    test_transform = transforms.Compose([transforms.ToTensor()])
  elif dataset == "tinyimagenet":
    train_transform = transforms.Compose([
        transforms.RandomCrop(64, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    test_transform = transforms.Compose([transforms.ToTensor()])

  num_workers = 2
  if dataset == "cifar10":
    train_dataset = datasets.CIFAR10(dir_,
                                     train=True,
                                     transform=train_transform,
                                     download=True)
    test_dataset = datasets.CIFAR10(dir_,
                                    train=False,
                                    transform=test_transform,
                                    download=True)

  elif dataset == "svhn":
    train_dataset = datasets.SVHN(dir_,
                                  split='train',
                                  transform=train_transform,
                                  download=True)
    test_dataset = datasets.SVHN(dir_,
                                 split='test',
                                 transform=test_transform,
                                 download=True)

  elif dataset == "tinyimagenet":
    train_dataset = torchvision.datasets.ImageFolder(root=dir_ + '/train',
                                                     transform=train_transform)
    test_dataset = torchvision.datasets.ImageFolder(root=dir_ + '/val',
                                                    transform=test_transform)

  train_loader = torch.utils.data.DataLoader(
      dataset=train_dataset,
      batch_size=batch_size,
      shuffle=True,
      pin_memory=True,
      num_workers=num_workers,
  )
  test_loader = torch.utils.data.DataLoader(
      dataset=test_dataset,
      batch_size=batch_size,
      shuffle=False,
      pin_memory=True,
      num_workers=num_workers,
  )
  return train_loader, test_loader


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', default=128, type=int)
  parser.add_argument('--data-dir', default='../cifar-data', type=str)
  parser.add_argument('--dataset', default='cifar10', type=str)
  parser.add_argument('--epochs', default=30, type=int)
  parser.add_argument('--n_classes', default=10, type=int)
  parser.add_argument('--lr-max', default=0.21, type=float)
  parser.add_argument('--attack',
                      default='pgd',
                      type=str,
                      choices=['pgd', 'fgsm', 'free', 'none'])
  parser.add_argument('--attack_type', default='none', type=str)
  parser.add_argument('--norm', default='l1', type=str)
  parser.add_argument('--epsilon', default=8, type=int)
  parser.add_argument('--attack-iters', default=8, type=int)
  parser.add_argument('--restarts', default=1, type=int)
  parser.add_argument('--pgd-alpha', default=2, type=int)
  parser.add_argument('--fname', default='cifar_model_free1', type=str)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--width-factor', default=10, type=int)
  parser.add_argument('--model', default='WideResNet')
  parser.add_argument('--js_weight', default=16, type=float)
  return parser.parse_args()


def main():
  args = get_args()
  logger.info(args)

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  args.data_dir = args.dataset + "-data"
  if args.dataset == "tinyimagenet":
    args.n_classes = 200
  else:
    args.n_classes = 10

  start_start_time = time.time()
  train_loader, test_loader = get_loaders(args.data_dir, args.batch_size,
                                          args.dataset)

  epsilon = (args.epsilon / 255.) / std
  pgd_alpha = (args.pgd_alpha / 255.) / std

  if args.model == 'WideResNet':
    model = WideResNet(28, 10, widen_factor=args.width_factor, dropRate=0.0)
  elif args.model == 'resnet50':
    model = ResNet50()
  else:
    raise ValueError("Unknown model")

  model = torch.nn.DataParallel(model).cuda()

  model.apply(initialize_weights)
  model.train()
  opt = torch.optim.SGD(model.module.params(),
                        lr=args.lr_max,
                        momentum=0.9,
                        weight_decay=5e-4)
  criterion = nn.CrossEntropyLoss()
  epochs = args.epochs

  lr_schedule = lambda t: np.interp(
        [t], [0, args.epochs * 2 // 5, args.epochs], [0, args.lr_max, 0])[0]

  logger.info('Epoch \t Time \t LR \t \t Train Loss \t Train Acc')

  for epoch in range(epochs):
    start_time = time.time()
    train_loss = 0
    train_acc = 0
    train_n = 0
    #为什么不用对训练集进行训练 只是对训练集产生的扰动图像训练？
    for i, (X, y) in enumerate(train_loader):
      X, y = X.cuda(), y.cuda()
      lr = lr_schedule(epoch + (i + 1) / len(train_loader))
      opt.param_groups[0].update(lr=lr)

      if args.attack == 'pgd':
        if args.attack_type == "none":
          delta = attack_pgd(model, X, y, opt, args.norm, args.dataset)
        elif args.attack_type == "msd":
          delta = attack_msd(model, X, y, opt, args.dataset)
        elif args.attack_type == "random":
          norms_list = ["linf", "l1", "l2"]
          curr_norm = random.choices(norms_list)
          delta = attack_pgd(model, X, y, opt, curr_norm[0], args.dataset)
        elif args.attack_type == "max" or args.attack_type == "avg" or args.attack_type == "avg_loss":
          norms_list = ["linf", "l1", "l2"]
          delta_linf = attack_pgd(model, X, y, opt, norms_list[0],
                                  args.dataset)
          delta_l1 = attack_pgd(model, X, y, opt, norms_list[1], args.dataset)
          delta_l2 = attack_pgd(model, X, y, opt, norms_list[2], args.dataset)

      elif args.attack == 'none':
        delta = torch.zeros_like(X)

      if args.attack_type == "none" or args.attack_type == "random" or args.attack_type == "msd":
        output = model(clamp(X + delta[:X.size(0)], lower_limit, upper_limit))
        loss = criterion(output, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

      elif args.attack_type == "max":
        output_linf = model(
            clamp(X + delta_linf[:X.size(0)], lower_limit, upper_limit))
        output_l1 = model(
            clamp(X + delta_l1[:X.size(0)], lower_limit, upper_limit))
        output_l2 = model(
            clamp(X + delta_l2[:X.size(0)], lower_limit, upper_limit))
        batch_size = X.shape[0]
        loss_linf = nn.CrossEntropyLoss(reduction='none')(output_linf, y)
        loss_l1 = nn.CrossEntropyLoss(reduction='none')(output_l1, y)
        loss_l2 = nn.CrossEntropyLoss(reduction='none')(output_l2, y)

        delta_l1 = delta_l1.view(batch_size, 1, -1)
        delta_l2 = delta_l2.view(batch_size, 1, -1)
        delta_linf = delta_linf.view(batch_size, 1, -1)

        loss_list = [loss_l1, loss_l2, loss_linf]
        delta_list = [delta_l1, delta_l2, delta_linf]
        loss_arr = torch.stack(tuple(loss_list))
        delta_arr = torch.stack(tuple(delta_list))
        max_loss = loss_arr.max(dim=0)
        delta = delta_arr[max_loss[1], torch.arange(batch_size), 0]
        delta = delta.view(batch_size, 3, X.shape[2], X.shape[3])

        output = model(clamp(X + delta[:X.size(0)], lower_limit, upper_limit))
        loss = criterion(output, y)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

      elif args.attack_type == "avg_loss":
        output = model(
            clamp(X + delta_linf[:X.size(0)], lower_limit, upper_limit))
        loss_linf = criterion(output, y)
        output_l1 = model(
            clamp(X + delta_l1[:X.size(0)], lower_limit, upper_limit))
        loss_l1 = criterion(output_l1, y)
        output_l2 = model(
            clamp(X + delta_l2[:X.size(0)], lower_limit, upper_limit))
        loss_l2 = criterion(output_l2, y)
        loss = (loss_linf + loss_l1 + loss_l2) / 3

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

      train_loss += loss.item() * y.size(0)
      train_acc += (output.max(1)[1] == y).sum().item()
      train_n += y.size(0)

    best_state_dict = copy.deepcopy(model.state_dict())

    train_time = time.time()
    print('%d \t %.1f \t %.4f \t %.4f \t %.4f' %
          (epoch, (train_time - start_time) / 60, lr, train_loss / train_n,
           train_acc / train_n))
  torch.save(best_state_dict, args.fname+'l1' + '.pth')
  logger.info('Total train time: %.4f minutes',
              (train_time - start_start_time) / 60)


if __name__ == "__main__":
  main()
