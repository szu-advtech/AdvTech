import math
import numpy as np
import torchvision.models
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
import argparse, random
from functools import partial
import os
from CA_block import resnet18_pos_attention
from PC_module import VisionTransformer_POS
import re
from torchvision.transforms import Resize
from CASME3_Dataset import CASME3_4,CASME3_OF,CASME3_depth,CASME3_RGBD

torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)


def initialize_weight_goog(m, n=''):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1.0)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        fan_out = m.weight.size(0)  # fan-out
        fan_in = 0
        if 'routing_fn' in n:
            fan_in = m.weight.size(1)
        init_range = 1.0 / math.sqrt(fan_in + fan_out)
        m.weight.data.uniform_(-init_range, init_range)
        m.bias.data.zero_()

def criterion2(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred
    y_pred_neg = y_pred - y_true * 1e12
    y_pred_pos = y_pred - (1 - y_true) * 1e12
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat((y_pred_neg, zeros), dim=-1)
    y_pred_pos = torch.cat((y_pred_pos, zeros), dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return torch.mean(neg_loss + pos_loss)


class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()

        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=90*2, kernel_size=3, stride=2, padding=1, bias=False,groups=1),
            nn.BatchNorm2d(180),
            nn.ReLU(inplace=True),
            )
        self.pos = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=512, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            )
        ##Position Calibration Module(subbranch)
        self.vit_pos=VisionTransformer_POS(img_size=14,
        patch_size=1, embed_dim=512, depth=3, num_heads=4, mlp_ratio=2, qkv_bias=True,norm_layer=partial(nn.LayerNorm, eps=1e-6),drop_path_rate=0.3)
        self.resize=Resize([14,14])
        ##main branch consisting of CA blocks
        self.main_branch =resnet18_pos_attention(class_num=4)
        self.head1 = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(1 * 112 *112, 38,bias=False),)

        self.timeembed = nn.Parameter(torch.zeros(1, 4, 111, 111))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x1, x5, if_shuffle):
        ##onset:x1 apex:x5
        B = x1.shape[0]
        #x = x1[:, :3]
        #Position Calibration Module (subbranch)
        POS = self.vit_pos(self.resize(x1)).transpose(1, 2).view(B, 512, 14, 14)
        act = x5 - x1
        act = self.conv_act(act)
        #main branch and fusion
        out, _ = self.main_branch(act, POS)

        return out

def run_training(args):

    imagenet_pretrained = True

    if not imagenet_pretrained:
        for m in res18.modules():
            initialize_weight_goog(m)

    if args.pretrained:
        print("Loading pretrained weights...", args.pretrained)
        pretrained = torch.load(args.pretrained)
        pretrained_state_dict = pretrained['state_dict']
        model_state_dict = res18.state_dict()
        loaded_keys = 0
        total_keys = 0
        for key in pretrained_state_dict:
            if ((key == 'module.fc.weight') | (key == 'module.fc.bias')):
                pass
            else:
                model_state_dict[key] = pretrained_state_dict[key]
                total_keys += 1
                if key in model_state_dict:
                    loaded_keys += 1
        print("Loaded params num:", loaded_keys)
        print("Total params num:", total_keys)
        res18.load_state_dict(model_state_dict, strict=False)

    ### data normalization for both training set
    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),

        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),

    ])
    ### data augmentation for training set only
    data_transforms_norm = transforms.Compose([

        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(4),
        transforms.RandomCrop(224, padding=4),
    ])

    ### data normalization for both teating set
    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    if args.loss=='CE_loss':
        criterion = torch.nn.CrossEntropyLoss()
    #leave one subject out protocal
    elif args.loss=='weighted_loss':
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([2.26339846,0.25379067,0.62307036,0.8597405]))
    else:
        raise ValueError('loss error')
    criterion= criterion.cuda()
    val_now = 0
    num_sum = 0
    pos_pred_ALL = torch.zeros(4)
    pos_label_ALL = torch.zeros(4)
    TP_ALL = torch.zeros(4)


    sub = [1, 10, 11, 12, 13, 138, 139, 14, 142, 143, 144, 145, 146, 147, 148, 149, 15, 150, 152, 153, 154, 155, 156,
           157, 158, 159, 160, 161, 162, 163, 165, 166, 167, 169, 17, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179,
           180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 2, 200, 201, 202,
           203, 204, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 3, 39, 4, 40, 41, 42, 5, 6, 7, 77, 8,
           9]
    for subj in sub:

        if args.input == 'apex+of':
            train_dataset = CASME3_OF(args.raf_path, train=True, num_loso=subj, transform=data_transforms,
                                   basic_aug=True, transform_norm=data_transforms_norm)
            val_dataset = CASME3_OF(args.raf_path, train=False, num_loso=subj, transform=data_transforms_val)
        elif args.input == 'apex-onset':
            train_dataset = CASME3_4(args.raf_path, train=True, num_loso=subj, transform=data_transforms,
                                       basic_aug=True, transform_norm=data_transforms_norm)
            val_dataset = CASME3_4(args.raf_path, train=False, num_loso=subj, transform=data_transforms_val)
        elif args.input == 'depth':
            train_dataset = CASME3_depth(args.raf_path, train=True, num_loso=subj, transform=data_transforms,
                                   basic_aug=True, transform_norm=data_transforms_norm)
            val_dataset = CASME3_depth(args.raf_path, train=False, num_loso=subj, transform=data_transforms_val)
        elif args.input == 'RGBD':
            train_dataset = CASME3_RGBD(args.raf_path, train=True, num_loso=subj, transform=data_transforms,
                                   basic_aug=True, transform_norm=data_transforms_norm)
            val_dataset = CASME3_RGBD(args.raf_path, train=False, num_loso=subj, transform=data_transforms_val)
        else:
            raise ValueError("dataset error")
        if val_dataset.__len__() == 0:
            continue
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=args.batch_size,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=args.batch_size,
                                                 num_workers=args.workers,
                                                 shuffle=False,
                                                 pin_memory=True)
        print('num_sub', subj)
        print('Train set size:', train_dataset.__len__())
        print('Validation set size:', val_dataset.__len__())

        max_corr = 0
        max_f1 = 0
        max_pos_pred = torch.zeros(4)
        max_pos_label = torch.zeros(4)
        max_TP = torch.zeros(4)
        ##model initialization
        net_all = MMNet()
        net_all = nn.DataParallel(net_all).cuda()
        params_all = net_all.parameters()

        if args.optimizer == 'adam':
            optimizer_all = torch.optim.Adam(params_all, lr=args.lr, weight_decay=0.7)

        elif args.optimizer == 'adamW':
            optimizer_all = torch.optim.AdamW(params_all,lr=args.lr,weight_decay=0.7)

        elif args.optimizer == 'sgd':
            optimizer_all = torch.optim.SGD(params_all,lr=args.lr,weight_decay=0.7)

        else:
            raise ValueError("Optimizer not supported.")
        ##lr_decay
        scheduler_all = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=0.987)

        net_all = net_all.cuda()

        for i in range(args.epochs):
            running_loss = 0.0
            correct_sum = 0
            running_loss_MASK = 0.0
            correct_sum_MASK = 0
            iter_cnt = 0

            net_all.train()
            #train for every epoch
            for batch_i, (img_onset, img_apex, emo) in enumerate(tqdm(train_loader)):
                iter_cnt += 1

                img_onset = img_onset.cuda()
                img_apex = img_apex.cuda()
                emo = emo.cuda()

                ##train MMNet
                ALL = net_all(img_onset, img_apex, False)
                loss_all = criterion(ALL, emo)
                optimizer_all.zero_grad()
                loss_all.backward()
                optimizer_all.step()
                running_loss += loss_all
                _, predicts = torch.max(ALL, 1)
                correct_num = torch.eq(predicts, emo).sum()
                correct_sum += correct_num

            ## lr decay
            if i <= 50:
                scheduler_all.step()
            if i>=0:
                acc = correct_sum.float() / float(train_dataset.__len__())
                running_loss = running_loss / iter_cnt
                print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))

            pos_label = torch.zeros(4)
            pos_pred = torch.zeros(4)
            TP = torch.zeros(4)
            ##test
            with torch.no_grad():
                running_loss = 0.0
                iter_cnt = 0
                bingo_cnt = 0
                sample_cnt = 0
                pre_lab_all = []
                Y_test_all = []
                net_all.eval()
                # net_au.eval()
                for batch_i, (img_onset, img_apex, emo) in enumerate(val_loader):

                    img_onset = img_onset.cuda()
                    img_apex = img_apex.cuda()
                    emo = emo.cuda()

                    ##test
                    ALL = net_all(img_onset,  img_apex, False)

                    loss = criterion(ALL,emo)
                    running_loss += loss
                    iter_cnt += 1
                    _, predicts = torch.max(ALL, 1)
                    correct_num = torch.eq(predicts, emo)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += ALL.size(0)

                    for cls in range(4):

                        for element in predicts:
                            if element == cls:
                                pos_label[cls] = pos_label[cls] + 1
                        for element in emo:
                            if element == cls:
                                pos_pred[cls] = pos_pred[cls] + 1
                        for elementp, elementl in zip(predicts, emo):
                            if elementp == elementl and elementp == cls:
                                TP[cls] = TP[cls] + 1
                count = 0
                SUM_F1 = 0
                for index in range(4):
                    if pos_label[index] != 0 or pos_pred[index] != 0:
                        count = count + 1
                        SUM_F1 = SUM_F1 + 2 * TP[index] / (pos_pred[index] + pos_label[index])

                AVG_F1 = SUM_F1 / count

                running_loss = running_loss / iter_cnt
                acc = bingo_cnt.float() / float(sample_cnt)
                acc = np.around(acc.numpy(), 4)
                if bingo_cnt > max_corr:
                    max_corr = bingo_cnt
                if AVG_F1 >= max_f1:
                    max_f1 = AVG_F1
                    max_pos_label = pos_label
                    max_pos_pred = pos_pred
                    max_TP = TP
                print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f, F1-score:%.3f" % (i, acc, running_loss, AVG_F1))
                if acc == 1 and AVG_F1 == 1:
                    break
        num_sum = num_sum + max_corr
        pos_label_ALL = pos_label_ALL + max_pos_label
        pos_pred_ALL = pos_pred_ALL + max_pos_pred
        TP_ALL = TP_ALL + max_TP
        count = 0
        SUM_F1 = 0
        F1_list = []
        for index in range(4):
            if pos_label_ALL[index] != 0 or pos_pred_ALL[index] != 0:
                count = count + 1
                F1 = 2 * TP_ALL[index] / (pos_pred_ALL[index] + pos_label_ALL[index])
                F1_list.append(F1)
                SUM_F1 = SUM_F1 + F1

        F1_ALL = SUM_F1 / count
        val_now = val_now + val_dataset.__len__()
        print("[..........%s] correctnum:%d . zongshu:%d   " % (subj, max_corr, val_dataset.__len__()))
        print("[ALL_corr]: %d [ALL_val]: %d" % (num_sum, val_now))
        print("[F1_now]: %.4f [F1_ALL]: %.4f" % (max_f1, F1_ALL))
        print('F1_list',F1_list)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='/NAS_REMOTE/wangzihan', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--relabel_epoch', type=int, default=1000,
                        help='Relabeling samples on each mini-batch after 10(Default) epochs.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')


    parser.add_argument('--optimizer', type=str, default='adamW', help='Optimizer, adam or sgd.')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate .')
    parser.add_argument('--epochs', type=int, default=75, help='Total training epochs.')
    parser.add_argument('--gpu', type=str, default='0', help='gpu-id')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--input', type=str, default='apex-onset', help='input data type,apex-onset or apex+of or depth or RGBD')
    parser.add_argument('--loss',type=str,default='CE_loss',help='loss type, weighted_loss or CE_loss')
    return parser.parse_args()
if __name__ == "__main__":
    cfg = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu
    torch.cuda.manual_seed_all(cfg.seed)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)
    run_training(cfg)
    print('lr={},batchsize={},epoch={},optimizer={},seed={}, input:{}, loss:{}'.format(cfg.lr, cfg.batch_size, cfg.epochs, cfg.optimizer, cfg.seed, cfg.input,cfg.loss))
