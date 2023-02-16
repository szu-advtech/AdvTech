#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.9


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter

from options import args_parser
from update_ori import LocalUpdate

from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGG, VGG_kvasir, CovidNet
#from other_models import MobileNetV2

from utils_ori import get_dataset, average_weights, exp_details, weights2cpu, weights2gpu, average_weights_online, COVIDxDataset
from collections import OrderedDict
from torchvision.models import resnet50, resnext50_32x4d, mobilenet_v2
import pdb


if __name__ == '__main__':
    start_time = time.time()
    args = args_parser()
    exp_details(args)

    # define paths
    path_project = os.path.abspath('..')
    #path_project = os.path.abspath('..')
    writer = SummaryWriter('log_other/' + str(args.exp_id) +
                           "/" + args.dataset + "/" + str(args.num_local))
    logger = SummaryWriter('../logs')

    # print(args)

    if args.gpu == '0':
        device = 'cuda:0'
    if args.gpu == '1':
        device = 'cuda:1'

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # load dataset and user groups
    train_dataset, test_dataset, user_groups = get_dataset(args)

    # BUILD MODEL
    if args.model == 'cnn':
        # Convolutional neural netork
        if args.dataset == 'mnist':
            global_model = CNNMnist(args=args)
        elif args.dataset == 'fmnist':
            global_model = CNNFashion_Mnist(args=args)
        elif args.dataset == 'cifar':
            #global_model = CNNCifar(args=args)
            global_model = VGG('VGG11')
        elif args.dataset == "kvasir":
            global_model = VGG_kvasir('VGG11')
        elif args.dataset == "covid":
            global_model = CovidNet()

    elif args.model == "resnet50":
        global_model = resnet50(num_classes=3)
    elif args.model == "Mobile_Net":
        global_model = mobilenet_v2(num_classes=3)
    elif args.model == "resnetxt":
        global_model = resnext50_32x4d(num_classes=3)

    elif args.model == 'mlp':
        # Multi-layer preceptron
        img_size = train_dataset[0][0].shape
        len_in = 1
        for x in img_size:
            len_in *= x
            global_model = MLP(dim_in=len_in, dim_hidden=64,
                               dim_out=args.num_classes)
    else:
        exit('Error: unrecognized model')

    # copy weights
    global_weights = global_model.state_dict()

    if args.dataset == 'cifar':
        local_weights_names = [s for s in global_weights.keys(
        ) if s.startswith('classifier')]  # fc for temp

    if args.dataset == 'kvasir':
        local_weights_names = [s for s in global_weights.keys(
        ) if s.startswith('classifier')]  # fc for temp
    if args.dataset == "covid":

        if args.model == "CovidNet":
            if args.num_local == 0:
                local_weights_names = []
            if args.num_local == 1:
                local_weights_names = [s for s in global_weights.keys(
                ) if s.startswith('classifier') or s.startswith("fc")]

        if args.model == "resnet50":
            if args.num_local == 0:
                local_weights_names = []
            if args.num_local == 1:
                local_weights_names = [
                    s for s in global_weights.keys() if s.startswith('fc')]

        if args.model == "Mobile_Net":
            if args.num_local == 0:
                local_weights_names = []
            if args.num_local == 1:
                local_weights_names = [
                    s for s in global_weights.keys() if s.startswith('classifier')]

        if args.model == "resnetxt":
            if args.num_local == 0:
                local_weights_names = []
            if args.num_local == 1:
                local_weights_names = [
                    s for s in global_weights.keys() if s.startswith('fc')]

    original_weights = copy.deepcopy(global_weights)
    local_original_weights = OrderedDict()
    for name in local_weights_names:
        local_original_weights[name] = original_weights[name]

    avg_we_cpu = weights2cpu(global_weights)

    refer_weights = dict()  # to record the local weights for each client

    # Training
    train_loss, train_accuracy, test_accuracy, train_local_test_loss = [], [], [], []
    val_acc_list, net_list = [], []
    cv_loss, cv_acc = [], []
    print_every = 5
    val_loss_pre = 0
    best_train_acc = 0
    best_test_acc = 0
    best_train_confusion_MA = []
    best_test_confusion_MA = []

    for epoch in tqdm(range(args.epochs)):

        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        print(idxs_users)

        counter = 0

        for idx in idxs_users:

            counter += 1

            temp_model = copy.deepcopy(global_model)

            temp_local_weights = temp_model.state_dict()
            if idx in refer_weights:
                pre_weights = refer_weights[idx]
                temp_local_weights.update(pre_weights)

            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[idx], logger=logger)

            temp_model.to(device)
            temp_model.train()

            w, loss = local_model.update_weights(
                model=temp_model, global_round=epoch)

            del local_model

            w_cpu = weights2cpu(w)
            temp_model.to("cpu")
            local_weights.append(copy.deepcopy(w))
            del temp_model, w
            torch.cuda.empty_cache()

            updated_local_weights = OrderedDict()
            for name in local_weights_names:
                updated_local_weights[name] = w_cpu[name]
            refer_weights[idx] = updated_local_weights
            del updated_local_weights

            if counter == 1:
                avg_we_cpu = w_cpu
            else:
                avg_we_cpu = average_weights_online(avg_we_cpu, w_cpu, counter)

            local_losses.append(copy.deepcopy(loss))

            torch.cuda.empty_cache()

        # update global weights
        global_weights = avg_we_cpu

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg)

        # Calculate avg training accuracy over all users at every epoch
        list_acc, list_loss, list_confusion_MA = [], [], []
        for c in range(args.num_users):
            local_model = LocalUpdate(args=args, dataset=train_dataset,
                                      idxs=user_groups[c], logger=logger)

            temp_model = copy.deepcopy(global_model)
            temp_global_weights = temp_model.state_dict()
            if c in refer_weights:
                temp_global_weights.update(refer_weights[c])
            else:
                temp_global_weights.update(local_original_weights)
            temp_model.load_state_dict(temp_global_weights)
            temp_model.to(device)
            temp_model.eval()

            #acc, loss = local_model.inference(model=global_model)
            acc, loss, confusion_MA = local_model.inference(
                model=temp_model, idx_epoch=epoch)
            list_acc.append(acc)
            list_loss.append(loss)
            list_confusion_MA.append(confusion_MA)
            temp_model.to("cpu")

            del temp_model
        train_local_test_loss.append(sum(list_loss) / len(list_loss))
        train_accuracy.append(sum(list_acc) / len(list_acc))
        writer.add_scalar("local_test_accuracy/epoch",
                          sum(list_acc) / len(list_acc), epoch + 1)
        writer.add_scalar("local_test_loss/epoch",
                          sum(list_loss) / len(list_loss), epoch + 1)
        if train_accuracy[-1] > best_train_acc:
            best_train_acc = train_accuracy[-1]
            best_train_confusion_MA = list_confusion_MA

        # print global training loss after every 'i' rounds
        if (epoch+1) % 1 == 0:
            temp_model = copy.deepcopy(global_model)
            temp_model.to(device)
            temp_model.eval()
            test_acc, test_loss, test_confusion_MA = LocalUpdate.test_inference(
                args, temp_model, test_dataset, idx_epoch=epoch)
            temp_model.to("cpu")
            del temp_model
            test_accuracy.append(test_acc)
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_confusion_MA = test_confusion_MA

            print(f' \nAvg Training Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.mean(np.array(train_loss))}')
            print('Train Accuracy: {:.2f}% \n'.format(100*train_accuracy[-1]))
            writer.add_scalar("global_test_accuracy/epoch",
                              test_acc, epoch + 1)
            print(list_acc, train_accuracy, test_accuracy)
            print(best_train_acc)
            for c_confusion_MA in best_train_confusion_MA:
                print(c_confusion_MA)
            # print(best_train_confusion_MA)
            print(best_test_acc)
            print(best_test_confusion_MA)
            # pdb.set_trace()
        if (epoch+1) == args.epochs:
            #test_acc, test_loss = test_inference(args, global_model, test_dataset)
            print(f' \n Results after {epoch} global rounds of training:')
            print(
                "|---- Avg Train Accuracy: {:.2f}%".format(100 * train_accuracy[-1]))
            print(list_acc)
            print(best_train_acc)
            print(best_test_acc)
            array_list_acc = np.asarray(list_acc)
            np.save("temfig/" + str(args.exp_id) + "_local_test_loss_" +
                    str(args.num_local), train_local_test_loss)
            np.save("temfig/" + str(args.exp_id) +
                    "_list_acc_" + str(args.num_local), list_acc)
            np.save("temfig/" + str(args.exp_id) + "_test_accuracy_" +
                    str(args.num_local), test_accuracy)
            np.save("temfig/" + str(args.exp_id) + "_train_accuracy_" +
                    str(args.num_local), train_accuracy)
            np.save("temfig/" + str(args.exp_id) + "_train_confusion_MA_" +
                    str(args.num_local), best_train_confusion_MA)
            np.save("temfig/" + str(args.exp_id) + "_test_confusion_MA_" +
                    str(args.num_local), best_test_confusion_MA)

    print(f' \n Results after {args.epochs} global rounds of training:')
    print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # Saving the objects train_loss and train_accuracy:
    file_name = './save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
        format(args.dataset, args.model, args.epochs, args.frac, args.iid,
               args.local_ep, args.local_bs)

    with open(file_name, 'wb') as f:
        pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))
    writer.close()
