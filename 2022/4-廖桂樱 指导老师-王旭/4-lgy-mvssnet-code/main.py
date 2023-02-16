import argparse
import os
from train import train
from test import test

# Training settings
parser = argparse.ArgumentParser(description='forgery_detection')
parser.add_argument('--model', type=str, default='mvssnet',
                    help='model, (default: UNet)')
parser.add_argument('--mode', type=str, default='test')
parser.add_argument('--suffix', type=str, default='_casia2.0',
                    help='the specification suffix of the saved model, (default: None)')
parser.add_argument('--load', type=bool, default=False,
                    help='load the last weight path and keep training, (default: False)')
parser.add_argument('--test-num', type=int, default=-1,
                    help='load the last weight path and keep training, (default: False)')
parser.add_argument('--multi-gpu', type=str, default='2',
                    help='multiple gpu for training, (default: 3, 4)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of threads started by dataloader, (default: 4)')
parser.add_argument('--seed', type=int, default=2022,
                    help='random seed, (default: 2022)')
parser.add_argument('--root', type=str, default='/data/liaogy/Datasets/CASIA',
                   help='the dir of dataset')
parser.add_argument('--txt', type=str, default='casia2.txt',
                    help='The index file of the dataset')
parser.add_argument('--val_txt', type=str, default='casia1.txt',
                    help='The index file of the dataset')
parser.add_argument('--lr', type=float, default=1e-6,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--max-epoches', type=float, default=100,
                    help='learning rate (default: 1000)')
parser.add_argument('--bz', type=int, default=4,
                    help='batch size (default: 64)')
parser.add_argument('--size', type=float, default=(512, 512),
                    help='image size (default: (512, 512))')
parser.add_argument('--val-interval', type=int, default=1,
                    help='val interval (default: 50)')
parser.add_argument('--ratio', type=float, default=1,
                    help='ratio of train set to val set (default: 0.8)')
parser.add_argument('--weight-save-dir', type=str, default='/data/liaogy/weight',
                    help='the saved directory of weight file')
parser.add_argument('--result-save-dir', type=str, default='/data/liaogy/result',
                    help='the saved directory of weight file')
parser.add_argument('--flag', type=int, default=3)



if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.multi_gpu
    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
