import argparse
import logging
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0 , 3"
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import laplace, uniform
from torchvision import datasets, transforms

from torch_backend import *
from tqdm import tqdm
import random

from preact_resnet import resnet50 as ResNet50
from wideresnet import WideResNet
import foolbox as fb
import foolbox.attacks as fa
from autoattack import AutoAttack

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(asctime)s] - %(message)s',
                    datefmt='%Y/%m/%d %H:%M:%S',
                    level=logging.DEBUG)

cifar10_mean = (0.0, 0.0, 0.0)
cifar10_std = (1.0, 1.0, 1.0)
mu = torch.tensor(cifar10_mean).view(3, 1, 1).cuda()
std = torch.tensor(cifar10_std).view(3, 1, 1).cuda()
upper_limit = ((1 - mu) / std)
lower_limit = ((0 - mu) / std)


def clamp(X, lower_limit, upper_limit):
  return torch.max(torch.min(X, upper_limit), lower_limit)


def norms(Z):
  return Z.view(Z.shape[0], -1).norm(dim=1)[:, None, None, None]


def norms_p(Z, order):
  return Z.view(Z.shape[0], -1).norm(p=order, dim=1)[:, None, None, None]


def norms_l1(Z):
  return Z.view(Z.shape[0], -1).abs().sum(dim=1)[:, None, None, None]


def l1_dir_topk(grad, delta, X, k=20):
  X_curr = X + delta
  batch_size = X.shape[0]
  channels = X.shape[1]
  pix = X.shape[2]

  grad = grad.detach().cpu().numpy()
  abs_grad = np.abs(grad)
  sign = np.sign(grad)

  max_abs_grad = np.percentile(abs_grad, k, axis=(1, 2, 3), keepdims=True)
  tied_for_max = (abs_grad >= max_abs_grad).astype(np.float32)
  num_ties = np.sum(tied_for_max, (1, 2, 3), keepdims=True)
  optimal_perturbation = sign * tied_for_max / num_ties

  optimal_perturbation = torch.from_numpy(optimal_perturbation).to(device)
  return optimal_perturbation.view(batch_size, channels, pix, pix)


def proj_l1ball(x, epsilon=12, device="cuda:1"):
  assert epsilon > 0
  u = x.abs()
  if (u.sum(dim=(1, 2, 3)) <= epsilon).all():
    return x
  y = proj_simplex(u, s=epsilon, device=device)
  y = y.view(-1, 3, x.shape[2], x.shape[3])
  y *= x.sign()
  return y


def proj_simplex(v, s=1, device="cuda:1"):
  assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
  batch_size = v.shape[0]
  u = v.view(batch_size, 1, -1)
  n = u.shape[2]
  u, indices = torch.sort(u, descending=True)
  cssv = u.cumsum(dim=2)
  vec = u * torch.arange(1, n + 1).float().to(device)
  comp = (vec > (cssv - s)).half()
  u = comp.cumsum(dim=2)
  w = (comp - 1).cumsum(dim=2)
  u = u + w
  rho = torch.argmax(u, dim=2)
  rho = rho.view(batch_size)
  c = torch.HalfTensor([cssv[i, 0, rho[i]]
                        for i in range(cssv.shape[0])]).to(device)
  c = c - s
  theta = torch.div(c.float(), (rho.float() + 1))
  theta = theta.view(batch_size, 1, 1, 1)
  w = (v.float() - theta).clamp(min=0)
  return w


def attack_pgd(model, X, y, norm, dataset, restarts=1, version=0):
  max_loss = torch.zeros(y.shape[0]).cuda()
  max_delta = torch.zeros_like(X).cuda()
  for _ in range(restarts):
    delta = torch.zeros_like(X).cuda()
    if norm == "linf":
      if dataset != "tinyimagenet":
        epsilon = (8 / 255.) / std
      else:
        epsilon = (4 / 255.) / std
      attack_iters = 50
      alpha = (1. / 255.) / std
      delta[:, 0, :, :].uniform_(-epsilon[0][0][0].item(),
                                 epsilon[0][0][0].item())
      delta[:, 1, :, :].uniform_(-epsilon[1][0][0].item(),
                                 epsilon[1][0][0].item())
      delta[:, 2, :, :].uniform_(-epsilon[2][0][0].item(),
                                 epsilon[2][0][0].item())
    elif norm == "l2":
      if dataset != "tinyimagenet":
        epsilon = (128. / 255.) / std
      else:
        epsilon = (80 / 255.) / std
      attack_iters = 50
      alpha = (25. / 255.) / std
      delta = torch.rand_like(X, requires_grad=True)
      delta.data *= (2.0 * delta.data - 1.0) * epsilon
      delta.data /= norms_p(
        delta.detach(), 2.0).clamp(min=epsilon.detach().cpu().numpy()[0][0][0])
    elif norm == "l1":
      epsilon = (2000 / 255.) / std
      attack_iters = 100
      alpha = (255. / 255.) / std
      ini = laplace.Laplace(loc=delta.new_tensor(0), scale=delta.new_tensor(1))
      delta.data = ini.sample(delta.data.shape)
      delta.data = (2.0 * delta.data - 1.0) * epsilon
      delta.data /= norms_l1(delta.detach()).clamp(min=epsilon.detach().cpu().numpy()[0][0][0])
    delta.requires_grad = True
    for _ in range(attack_iters):
      output = model(X + delta)
      incorrect = output.max(1)[1] != y
      correct = (~incorrect).unsqueeze(1).unsqueeze(1).unsqueeze(1).half()
      correct = 1.0 if version == 0 else correct
      loss = F.cross_entropy(output, y)
      loss.backward()
      grad = delta.grad.detach()
      if norm == "linf":
        delta.data = clamp(delta.data + correct * alpha * torch.sign(grad),
                           -epsilon, epsilon)
      elif norm == "l2":
        delta.data = delta.data + correct * alpha * grad / norms(grad)
        delta.data *= epsilon / norms_p(delta.detach(), 2.0).clamp(
          min=epsilon.detach().cpu().numpy()[0][0][0])
      elif norm == "l1":
        k = 99
        delta.data = delta.data + correct * alpha * l1_dir_topk(
            grad, delta.data, X, k)
        if (norms_l1(delta) > epsilon).any():
          delta.data = proj_l1ball(delta.data,
                               epsilon=epsilon.detach().cpu().numpy()[0][0][0],
                               device=device)
      delta.data = clamp(delta.data, lower_limit - X, upper_limit - X)
      delta.grad.zero_()
    all_loss = F.cross_entropy(model(X + delta), y, reduction='none')
    
  return delta.detach()


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
      num_workers=2,
  )
  return train_loader, test_loader


def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('--batch-size', default=26, type=int)
  parser.add_argument('--data-dir', default='../data/cifar-data', type=str)
  parser.add_argument('--dataset', default='cifar10', type=str)
  parser.add_argument('--attack',
                      default='pgd',
                      type=str,
                      choices=['pgd', 'fgsm', 'ddn', 'none'])
  parser.add_argument('--attack_lib', default='custom', type=str)
  parser.add_argument('--norm', default='linf', type=str)
  parser.add_argument('--epsilon', default=8, type=float)
  parser.add_argument('--attack-iters', default=200, type=int)
  parser.add_argument('--n_classes', default=10, type=int)
  parser.add_argument('--alpha', default=1, type=int)
  parser.add_argument('--restarts', default=1, type=int)
  parser.add_argument('--seed', default=0, type=int)
  parser.add_argument('--width-factor', default=10, type=int)
  parser.add_argument('--model', default='WideResNet')
  parser.add_argument('--model_type', default='random')
  parser.add_argument('--use_cal', default=True, type=bool)
  return parser.parse_args()


def get_attack(fmodel, attack, init_attack=None):
  # L0
  if attack == 'SAPA':
    A = fa.SaltAndPepperNoiseAttack()
  elif attack == 'EAD':
    A = fa.EADAttack(decision_rule='L1', binary_search_steps=5, steps=1000)
  elif attack == "PGDL1":
    A = fa.SparseL1DescentAttack()

  # L2
  elif 'PGDL2' in attack:
    A = fa.L2PGD()
  elif attack == 'AGNA':

    A = fa.L2RepeatedAdditiveGaussianNoiseAttack()
  elif attack == "CWL2":
    A = fa.L2CarliniWagnerAttack(binary_search_steps=5, steps=200)
  elif attack == "BBL2":
    A = fa.L2BrendelBethgeAttack(init_attack=init_attack)

  # L inf
  elif 'FGSM' in attack and not 'IFGSM' in attack:
    A = fa.FGSM()
  elif 'PGDLinf' in attack:
    A = fa.LinfPGD()
  elif 'MIM' in attack:
    A = fa.MomentumIterativeAttack()
  elif attack == "BBLinf":
    A = fa.LinfinityBrendelBethgeAttack(init_attack=init_attack)
  else:
    raise Exception('Not implemented')
  return A


def main():
  args = get_args()
  logger.info(args)

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)

  args.data_dir = args.dataset + "-data"
  train_loader, test_loader = get_loaders(args.data_dir, args.batch_size,
                                          args.dataset)

  epsilon = (args.epsilon / 255.) / std
  alpha = (args.alpha / 255.) / std

  if args.model == 'resnet50':
    model = ResNet50().cuda()
  elif args.model == 'WideResNet':
    model = WideResNet(28, 10, widen_factor=args.width_factor, dropRate=0.0)
  else:
    raise ValueError("Unknown model")

  model = torch.nn.DataParallel(model).cuda()

  checkpoint = torch.load("zhuqingying/mng_ac.pth",
                          map_location='cuda:0')
  model.load_state_dict(checkpoint)
  model.eval()
  model.float()

  total_adv_loss = 0
  total_adv_acc = 0
  total_clean_loss = 0
  total_clean_acc = 0
  total = 0

  if args.attack_lib == 'custom':
    x_adv = []
    total = 0
    max_check = 1000
    output = np.ones((max_check + 1))
    for i, (X, y) in tqdm(enumerate(test_loader)):
      X, y = X.cuda(), y.cuda()
      if args.attack == 'pgd':
        image = X[0, :, :, :].view(1, 3, X.shape[2], X.shape[3])
        label = y[0].long().view(-1)
        delta = attack_pgd(model, image, label, args.norm, args.dataset)
      with torch.no_grad():
        output_adv = model(image + delta)
        output_clean = model(image)
        total_adv_acc += (output_adv.max(1)[1] == label).item()
        total_clean_acc += (output_clean.max(1)[1] == label).item()
        output[total] = (output_adv.max(1)[1] == label).cpu().numpy().astype(
            np.float32)
        total += 1

      if (total >= max_check):
        break

    print("Test Adversarial Acc: {} in {}.".format(total_adv_acc / total , args.norm))
    print('Test Clean Acc in mngupdate: %.4f' %(total_clean_acc / total))

  elif args.attack_lib == "foolbox":
    fmodel = fb.PyTorchModel(model, bounds=(0, 1), device=device)
    attacks_list = [
        "PGDLinf", "BBLinf", "PGDL1", "EAD", "SAPA", 'BBL2', 'CWL2', 'AGNA',
        'PGDL2'
    ]
    for j in range(len(attacks_list)):
      start = time.time()
      attack_name = attacks_list[j]
      if "Linf" in attack_name:
        if args.dataset == "tinyimagenet":
          epsilons = [4. / 255]
        else:
          epsilons = [8. / 255]
      elif "L2" in attack_name:
        if args.dataset == "tinyimagenet":
          epsilons = [80. / 255]
        else:
          epsilons = [128. / 255]
      elif "L1" in attack_name:
        epsilons = [2000. / 255]
      total = 0
      robust_accuracy = 0
      total_adv_acc = 0
      max_check = 1000
      output = np.ones((max_check + 1))
      images, labels = iter(test_loader).next()
      images, labels = images.cuda(), labels.cuda()
      batches = [(images[:10], labels[:10])]
      init_attack = fb.attacks.DatasetAttack()
      init_attack.feed(fmodel, batches[0][0])  # feed 1st batch of inputs
      if args.dataset != "tinyimagenet":
        attack = get_attack(fmodel, attacks_list[j], init_attack)
      else:
        attack = get_attack(fmodel, attacks_list[j])

      for i, (X, y) in tqdm(enumerate(test_loader)):
        X, y = X.cuda(), y.cuda()
        image = X[0, :, :, :].view(1, 3, X.shape[2], X.shape[3])
        label = y[0].long().view(-1)
        try:
          advs, _, success = attack(fmodel, image, label, epsilons=epsilons)
          success = success.cpu().numpy().astype(np.float32)
          robust_accuracy += 1 - success.mean(axis=-1)
          output[total] = 1 - success.mean(axis=-1)
          total += 1
          output_adv = model(advs[0])
          # make_model_diagrams(output_adv, label)
        except Exception as e:
          output[total] = 1.0
          print("assertion error", e)
          total += 1
          continue

        if (total >= max_check):
          break

      print("Robust accuracy for attack %s: %.4f %.4f" %
            (attacks_list[j], robust_accuracy / total, total))

  elif args.attack_lib == "autoattack":
    if args.norm == "linf":
      if args.dataset == "tinyimagenet":
        epsilon = 4. / 255
      else:
        epsilon = 8. / 255
      adversary = AutoAttack(model,
                             norm='Linf',
                             eps=epsilon,
                             version='standard')
    elif args.norm == "l2":
      if args.dataset == "tinyimagenet":
        epsilon = 80. / 255
      else:
        epsilon = 128. / 255     
      adversary = AutoAttack(model, norm='L2', eps=epsilon, version='standard')
    max_check = 1000
    output = np.ones((max_check + 1))
    total = 0
    for i, (X, y) in tqdm(enumerate(test_loader)):
      X, y = X.cuda(), y.cuda()
      image = X[0, :, :, :].view(1, 3, X.shape[2], X.shape[3])
      label = y[0].long().view(-1)
      adv_X = adversary.run_standard_evaluation(image,
                                                label,
                                                bs=image.shape[0])
      output_adv = model(adv_X)
      loss_adv = F.cross_entropy(output_adv, label)
      total_adv_acc += (output_adv.max(1)[1] == label).item()
      output[total] = (output_adv.max(1)[1] == label).cpu().numpy().astype(
          np.float32)
      total += 1

      if (total >= max_check):
        break
    print('Test Adversarial Acc: %.4f' % (total_adv_acc / total))


if __name__ == "__main__":
  main()
