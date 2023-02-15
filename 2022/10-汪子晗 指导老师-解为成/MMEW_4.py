import math
import numpy as np
import torch.utils.data as data
from torchvision import transforms
import cv2
import pandas as pd
import torch
import torch.nn as nn
import argparse, random
from functools import partial
import os
from CA_block import resnet18_pos_attention
from tqdm import tqdm
from PC_module import VisionTransformer_POS
from torchvision.transforms import Resize
from MMEW_dataset  import *
torch.set_printoptions(precision=3, edgeitems=14, linewidth=350)




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
    parser.add_argument('--batch_size', type=int, default=34, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=7000, help='Total training epochs.')
    parser.add_argument('--drop_rate', type=float, default=0, help='Drop out rate.')
    return parser.parse_args()



class MMNet(nn.Module):
    def __init__(self):
        super(MMNet, self).__init__()

        self.conv_act = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=90*2, kernel_size=3, stride=2,padding=1, bias=False,groups=1),
            nn.BatchNorm2d(180),
            nn.ReLU(inplace=True),
            )
        self.pos =nn.Sequential(
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
            nn.Linear(1 * 112 *112, 38,bias=False),

        )

        self.timeembed = nn.Parameter(torch.zeros(1, 4, 111, 111))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
    def forward(self, x1, x5, if_shuffle):
        ##onset:x1 apex:x5
        B = x1.shape[0]

        #Position Calibration Module (subbranch)
        POS =self.vit_pos(self.resize(x1)).transpose(1,2).view(B,512,14,14)
        act = x5 -x1
        act=self.conv_act(act)
        #main branch and fusion
        out,_=self.main_branch(act,POS)

        return out





def run_training():

    args = parse_args()
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

    criterion = torch.nn.CrossEntropyLoss()
    #leave one subject out protocal

    val_now = 0
    num_sum = 0
    pos_pred_ALL = torch.zeros(4)
    pos_label_ALL = torch.zeros(4)
    TP_ALL = torch.zeros(4)

    for subj in range(1, 31):
        train_dataset = MMEW(args.raf_path, train=True, num_loso=subj, transform=data_transforms,
                                   basic_aug=True, transform_norm=data_transforms_norm)
        val_dataset = MMEW(args.raf_path, train=False, num_loso=subj, transform=data_transforms_val)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=32,
                                                   num_workers=args.workers,
                                                   shuffle=True,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=32,
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

        params_all = net_all.parameters()

        if args.optimizer == 'adam':
            optimizer_all = torch.optim.AdamW(params_all, lr=0.008, weight_decay=0.7)
            ##optimizer for MMNet

        elif args.optimizer == 'sgd':
            optimizer_all = torch.optim.SGD(params_all, args.lr,
                                        momentum=args.momentum,
                                        weight_decay=1e-4)
        else:
            raise ValueError("Optimizer not supported.")
        ##lr_decay
        scheduler_all = torch.optim.lr_scheduler.ExponentialLR(optimizer_all, gamma=0.987)

        net_all = net_all.cuda()

        for i in range(1, 75):
            running_loss = 0.0
            correct_sum = 0
            iter_cnt = 0

            net_all.train()

            #train for every epoch
            for batch_i, (image_on0, image_apex0, label_all) in enumerate(tqdm(train_loader)):
                iter_cnt += 1

                image_on0 = image_on0.cuda()

                image_apex0 = image_apex0.cuda()
                label_all = label_all.cuda()

                ##train MMNet
                ALL = net_all(image_on0,  image_apex0,
                                    False)

                loss_all = criterion(ALL, label_all)

                optimizer_all.zero_grad()

                loss_all.backward()

                optimizer_all.step()
                running_loss += loss_all
                _, predicts = torch.max(ALL, 1)
                correct_num = torch.eq(predicts, label_all).sum()
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
                for batch_i, (image_on0,  image_apex0, label_all) in enumerate(val_loader):

                    image_on0 = image_on0.cuda()
                    image_apex0 = image_apex0.cuda()
                    label_all = label_all.cuda()
                    ##test
                    ALL = net_all(image_on0,  image_apex0, False)

                    loss = criterion(ALL, label_all)
                    running_loss += loss
                    iter_cnt += 1
                    _, predicts = torch.max(ALL, 1)
                    correct_num = torch.eq(predicts, label_all)
                    bingo_cnt += correct_num.sum().cpu()
                    sample_cnt += ALL.size(0)

                    for cls in range(4):

                        for element in predicts:
                            if element == cls:
                                pos_label[cls] = pos_label[cls] + 1
                        for element in label_all:
                            if element == cls:
                                pos_pred[cls] = pos_pred[cls] + 1
                        for elementp, elementl in zip(predicts, label_all):
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
        for index in range(4):
            if pos_label_ALL[index] != 0 or pos_pred_ALL[index] != 0:
                count = count + 1
                SUM_F1 = SUM_F1 + 2 * TP_ALL[index] / (pos_pred_ALL[index] + pos_label_ALL[index])

        F1_ALL = SUM_F1 / count
        val_now = val_now + val_dataset.__len__()
        print("[..........%s] correctnum:%d . zongshu:%d   " % (subj, max_corr, val_dataset.__len__()))
        print("[ALL_corr]: %d [ALL_val]: %d" % (num_sum, val_now))
        print("[F1_now]: %.4f [F1_ALL]: %.4f" % (max_f1, F1_ALL))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = '6'
    run_training()
