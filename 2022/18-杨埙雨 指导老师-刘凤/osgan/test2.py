import os
import sys
import time
import argparse
import math
import matplotlib.pyplot as plt
import  torchvision.utils as vutils
import numpy as np
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
savepath = "./result/"
args = './cfgs/dcgan/dcgan_asymmetric_two_stage_celeba.yml'
cfg = load_yaml(args)
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
generator = get_model('{}_generator'.format(cfg['method']), zdim=cfg['zdim'],
                          num_channel=cfg['num_channel'])
netG = generator.cpu()




netG.load_state_dict(torch.load("./ckpt/dcgan_two_stage_celeba_asymmetric/generator-epoch030.pth",map_location=lambda  storage,loc: storage))
netG.eval()

torch.no_grad()

noise = torch.randn((64, cfg['zdim'], 1, 1)).cpu()
fake = netG(noise).detach().cpu()
vutils.save_image(fake.detach()[0:64],
					'%s/fake_samples.png' % (savepath+'dcgan_two/'), normalize=True)

