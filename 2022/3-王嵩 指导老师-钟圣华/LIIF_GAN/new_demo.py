import argparse
import math
import os
from PIL import Image

import torch
from torchvision import transforms

import models
from utils import make_coord
from test import batched_predict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='input.png')
    parser.add_argument('--model')
    parser.add_argument('--resolution', type=float)
    parser.add_argument('--output', default='output.png')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    img = transforms.ToTensor()(Image.open(args.input).convert('RGB')).cuda()

    model = models.make(torch.load(args.model)['model'], load_sd=True).cuda()
    print(img.shape)
    h = math.floor(img.shape[1]*args.resolution)
    w = math.floor(img.shape[2]*args.resolution)
    print('h:', h)
    print('w:', w)
    coord = make_coord((h, w)).cuda()  # H*W,2
    cell = torch.ones_like(coord)
    cell[:, 0] *= 2 / h  # 1-(-1)/h
    cell[:, 1] *= 2 / w

    # img --> 1,3,H,W 类似归一化Normalize
    # coord --> 1,H*w,2
    # cell --> 1,H*w,2
    inp_sub = torch.FloatTensor([[[0.422369]], [[0.426868]], [[0.836872]]]).cuda()
    # inp_sub = torch.FloatTensor([[[0]], [[0]], [[0]]]).cuda()
    inp_div = torch.FloatTensor([1]).cuda()
    pred = batched_predict(model, ((img - inp_sub) / inp_div).cuda().unsqueeze(0),
        coord.unsqueeze(0), cell.unsqueeze(0), bsize=30000)[0]
    pred = pred.view(h, w, 3).permute(2, 0, 1)
    pred = (pred * inp_div + inp_sub).clamp(0, 1).cpu()
    transforms.ToPILImage()(pred).save(args.output)
