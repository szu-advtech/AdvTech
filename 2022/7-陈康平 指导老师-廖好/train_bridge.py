# -*- coding: utf-8 -*-
# file: train.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2018. All Rights Reserved.
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import logging
import argparse
import math
# import os
import sys
from time import strftime, localtime, time
import random
import numpy
from evaluation import *

from pytorch_transformers import BertModel

from sklearn import metrics
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
# from data_utils import build_tokenizer, build_embedding_matrix, Tokenizer4Bert

from utils import ABSADataset

from models.decnn_base import DECNN

print('Model is decnn_base')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        'word index => word2id.txt'
        'word embedding => word_embedding.npy'
        if opt.reuse_embedding == 0:
            raise ValueError
        elif opt.reuse_embedding == 1:
            print('Reuse Word Dictionary & Embedding')
            with open('./data/word2id.txt', 'r', encoding='utf-8') as f:
                word_dict = eval(f.read())
            w2v_global = np.load('./data/cross_embedding.npy')
        # sys.exit()
        self.model = opt.model_class(w2v_global, opt).to(opt.device)

        if opt.use_unlabel == 1:
            self.trainset = ABSADataset('train', opt.dataset_file['train'], word_dict, opt, opt.dataset_file['unlabel'])
        else:
            self.trainset = ABSADataset('train', opt.dataset_file['train'], word_dict, opt)
        self.testset = ABSADataset('test', opt.dataset_file['test'], word_dict, opt)
        # assert 0 <= opt.valset_ratio < 1
        if opt.valset_num > 0:
            # valset_len = int(len(self.trainset) * opt.valset_ratio)
            # valset_len = 150
            self.trainset, self.valset = random_split(self.trainset, (len(self.trainset)- opt.valset_num, opt.valset_num))
        else:
            self.valset = self.testset

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(device=opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('> training arguments:')
        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    def _reset_params(self):
        # for child in self.model.children():
        #     if type(child) != BertModel:  # skip bert params
        #         for p in child.parameters():
        #             if p.requires_grad:
        #                 if len(p.shape) > 1:
        #                     stdv = 1. / math.sqrt(p.shape[0])
        #                     torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        #                 else:
        #                     stdv = 1. / math.sqrt(p.shape[0])
        #                     torch.nn.init.uniform_(p, a=-stdv, b=stdv)
        for child in self.model.children():
            # print('child:', child)
            pass
        for name, p in self.model.named_parameters():
            if 'bert' not in name:
                if p.requires_grad:
                    if len(p.shape) > 1:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)
                    else:
                        stdv = 1. / math.sqrt(p.shape[0])
                        torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    def _train(self, criterion, optimizer, train_data_loader, test_data_loader, dev_data_loader, is_training):
        max_dev_metric = 0.
        min_dev_loss = 1000.
        global_step = 0
        path = None
        aspect_f1_list, opinion_f1_list, sentiment_acc_list, sentiment_f1_list, ABSA_f1_list = list(), list(), list(), list(), list()
        dev_metric_list, dev_loss_list = list(), list()

        for epoch in range(self.opt.num_epoch):
            tau_now = np.maximum(1. * np.exp(-0.03 * epoch), 0.1)
            'TRAIN'
            epoch_start = time()
            n_correct, n_total, loss_total = 0, 0, 0
            aspect_loss_total, opinion_loss_total, domain_loss_total, sentiment_loss_total, reg_loss_total = 0., 0., 0., 0., 0.
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                for process in range(2): # Alternative Training
                    global_step += 1

                    # len_dataloader = len(train_data_loader)//2
                    # p = float(i_batch + epoch * len_dataloader) / self.opt.num_epoch / len_dataloader
                    # alpha = 2. / (1. + np.exp(-10 * p)) - 1
                    alpha = 0.1

                    # clear gradient accumulators
                    optimizer.zero_grad()

                    inputs = [sample_batched[col].to(self.opt.device) for col in ['x', 'mask', 'dep', 'pos', 'adj', 'lmwords', 'lmprobs']]
                    # outputs = torch.clamp(self.model(inputs, basedata).view(-1, self.opt.class_num), -1e-5, 1.)
                    # aspect_y = sample_batched['aspect_y'].to(self.opt.device).view(-1, self.opt.class_num)
                    aspect_y = sample_batched['aspect_y'].to(self.opt.device)
                    # opinion_y = sample_batched['opinion_y'].to(self.opt.device)
                    domain_y = sample_batched['domain_y'].to(self.opt.device)

                    aspect_outputs, domain_outputs = self.model(inputs, alpha=alpha, is_training=True)

                    aspect_outputs = torch.clamp(aspect_outputs, 1e-5, 1.)
                    domain_outputs = torch.clamp(domain_outputs, 1e-5, 1.)

                    length = torch.sum(inputs[1])
                    batch_size = (inputs[1]).shape[0]
                    # loss = criterion(outputs, aspect_y)
                    aspect_loss = torch.sum(-1 * torch.log(aspect_outputs) * aspect_y.float()) / (length + 1e-6)
                    opinion_loss = torch.tensor(0.)
                    domain_loss = torch.sum(-1 * torch.log(domain_outputs) * domain_y.float()) / (batch_size + 1e-6)
                    sentiment_loss = torch.tensor(0.)
                    reg_loss = torch.tensor(0.)

                    if self.opt.use_unlabel == 1:
                        loss = aspect_loss  + domain_loss
                    else:
                        loss = aspect_loss

                    'alternative backward'
                    if process == 0 and self.opt.use_unlabel == 1:
                        domain_loss.backward()
                        optimizer.step()
                    elif process == 1:
                        aspect_loss.backward()
                        optimizer.step()

                    n_total += length
                    loss_total += loss.item() * length
                    aspect_loss_total += aspect_loss.item() * length
                    opinion_loss_total += opinion_loss.item() * length
                    domain_loss_total += domain_loss.item() * batch_size
                    sentiment_loss_total += sentiment_loss.item() * length
                    reg_loss_total += reg_loss.item() * length

            train_loss = loss_total / n_total
            train_aspect_loss = aspect_loss_total / n_total
            train_opinion_loss = opinion_loss_total / n_total
            train_domain_loss = domain_loss_total / (len(train_data_loader.dataset) * 2)
            train_sentiment_loss = sentiment_loss_total / n_total
            train_reg_loss = reg_loss_total / n_total

            'DEV'
            dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1, \
            dev_loss, dev_aspect_loss, dev_opinion_loss, dev_sentiment_loss, dev_reg_loss = \
            self._evaluate_acc_f1(dev_data_loader, epoch, tau_now)
            dev_metric = dev_aspect_f1
            if epoch < 0:
                dev_metric_list.append(0.)
                dev_loss_list.append(1000.)
            else:
                dev_metric_list.append(dev_metric)
                dev_loss_list.append(dev_loss)

            save_indicator = 0
            if (dev_metric > max_dev_metric or dev_loss < min_dev_loss) and epoch >= 0:
                if dev_metric > max_dev_metric:
                    save_indicator = 1
                    max_dev_metric = dev_metric
                if dev_loss < min_dev_loss:
                    min_dev_loss = dev_loss

            'TEST'
            test_aspect_f1, test_opinion_f1, test_sentiment_acc, test_sentiment_f1, test_ABSA_f1, \
            test_loss, test_aspect_loss, test_opinion_loss, test_sentiment_loss, test_reg_loss = \
            self._evaluate_acc_f1(test_data_loader, epoch, tau_now)
            aspect_f1_list.append(test_aspect_f1)
            opinion_f1_list.append(test_opinion_f1)
            sentiment_acc_list.append(test_sentiment_acc)
            sentiment_f1_list.append(test_sentiment_f1)
            ABSA_f1_list.append(test_ABSA_f1)

            'EPOCH INFO'
            epoch_end = time()
            epoch_time = 'Epoch Time: {:.0f}m {:.0f}s'.format((epoch_end - epoch_start) // 60, (epoch_end - epoch_start) % 60)
            logger.info('\n{:-^80}'.format('Iter' + str(epoch)))
            logger.info('Train: final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, domain loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                        format(train_loss, train_aspect_loss, train_opinion_loss, train_domain_loss, train_sentiment_loss, train_reg_loss, global_step))
            logger.info('Dev:   final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                        format(dev_loss, dev_aspect_loss, dev_opinion_loss, dev_sentiment_loss, dev_reg_loss, global_step))
            logger.info('Dev:   aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                        .format(dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1))
            logger.info('Test:  aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                        .format(test_aspect_f1, test_opinion_f1, test_sentiment_acc, test_sentiment_f1, test_ABSA_f1))
            logger.info('Current Max Metrics Index : {} Current Min Loss Index : {} {} Tau : {:.2f}'
                        .format(dev_metric_list.index(max(dev_metric_list)), dev_loss_list.index(min(dev_loss_list)), epoch_time, tau_now))

            'SAVE CheckPoints'
            # if not os.path.exists('./state_dict/{}-to-{}'.format(self.opt.source, self.opt.target)):
            #     os.makedirs('./state_dict/{}-to-{}'.format(self.opt.source, self.opt.target))
            # if save_indicator == 1:
            #     path = 'state_dict/{}-to-{}/{}-ATE-F1-{:.4f}.pth'.format(self.opt.source, self.opt.target, self.opt.name, test_aspect_f1)
            #     torch.save(self.model.state_dict(), path)
            #     logger.info('>> Checkpoint Saved: {}'.format(path))



        'SUMMARY'
        logger.info('\n{:-^80}'.format('Mission Complete'))
        max_dev_index = dev_metric_list.index(max(dev_metric_list))
        logger.info('Dev Max Metrics Index: {}'.format(max_dev_index))
        logger.info('aspect f1={:.2f}, opinion f1={:.2f}, sentiment acc=={:.2f}, sentiment f1=={:.2f}, ABSA f1=={:.2f},'
                    .format(aspect_f1_list[max_dev_index]*100, opinion_f1_list[max_dev_index]*100, sentiment_acc_list[max_dev_index]*100,
                            sentiment_f1_list[max_dev_index]*100, ABSA_f1_list[max_dev_index]*100))

        min_dev_index = dev_loss_list.index(min(dev_loss_list))
        logger.info('Dev Min Loss Index: {}'.format(min_dev_index))
        logger.info('aspect f1={:.2f}, opinion f1={:.2f}, sentiment acc=={:.2f}, sentiment f1=={:.2f}, ABSA f1=={:.2f},'
                    .format(aspect_f1_list[min_dev_index]*100, opinion_f1_list[min_dev_index]*100, sentiment_acc_list[min_dev_index]*100,
                            sentiment_f1_list[min_dev_index]*100, ABSA_f1_list[min_dev_index]*100))

        return path

    def _evaluate_acc_f1(self, data_loader, epoch, tau_now):
        n_correct, n_total, loss_total = 0, 0, 0
        aspect_loss_total, opinion_loss_total, sentiment_loss_total, reg_loss_total = 0., 0., 0., 0.
        # t_aspect_y_all, t_outputs_all, t_mask_all = None, None, None
        t_aspect_y_all, t_aspect_outputs_all, t_opinion_y_all, t_opinion_outputs_all, t_mask_all = list(), list(), list(), list(), list()
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                inputs = [t_sample_batched[col].to(self.opt.device) for col in ['x', 'mask', 'dep', 'pos', 'adj', 'lmwords', 'lmprobs']]
                # t_outputs = torch.clamp(, 1e-5, 1.)
                t_aspect_y = t_sample_batched['aspect_y'].to(self.opt.device)

                t_aspect_outputs = self.model(inputs)
                t_aspect_outputs = torch.clamp(t_aspect_outputs, 1e-5, 1.)

                length = torch.sum(inputs[1])

                aspect_loss = torch.sum(-1 * torch.log(t_aspect_outputs) * t_aspect_y.float()) / length
                opinion_loss = torch.tensor(0.)
                sentiment_loss = torch.tensor(0.)
                reg_loss = torch.tensor(0.)
                loss = aspect_loss + opinion_loss + sentiment_loss + self.opt.l2_reg * reg_loss

                n_total += length
                loss_total += loss.item() * length
                aspect_loss_total += aspect_loss.item() * length
                opinion_loss_total += opinion_loss.item() * length
                sentiment_loss_total += sentiment_loss.item() * length
                reg_loss_total += reg_loss.item() * length


                t_aspect_y_all.extend(t_aspect_y.cpu().tolist())
                t_aspect_outputs_all.extend(t_aspect_outputs.cpu().tolist())
                t_mask_all.extend(inputs[1].cpu().tolist())
            t_loss = loss_total / n_total
            t_aspect_loss = aspect_loss_total / n_total
            t_opinion_loss = opinion_loss_total / n_total
            t_sentiment_loss = sentiment_loss_total / n_total
            t_reg_loss = reg_loss_total / n_total
        t_aspect_f1, t_opinion_f1, t_sentiment_acc, t_sentiment_f1, t_ABSA_f1 = get_metric(t_aspect_y_all, t_aspect_outputs_all,
                                           np.zeros_like(t_aspect_y_all), np.zeros_like(t_aspect_y_all),
                                           np.zeros_like(t_aspect_y_all),  np.zeros_like(t_aspect_y_all),
                                           t_mask_all, 1)
        # t_aspect_f1, t_opinion_f1, t_sentiment_acc, t_sentiment_f1, t_ABSA_f1 = 0.,0.,0.,0.,0.
        # # for case study
        # a_preds = np.array(t_outputs_all)
        # final_mask = np.array(t_mask_all)
        #
        # a_preds = np.argmax(a_preds, axis=-1)
        # # logger.info(np.shape(a_preds))
        # # logger.info(np.shape(final_mask))
        #
        # aspect_out = []
        #
        # for s_idx, sentence in enumerate(a_preds):
        #     aspect_iter = []
        #     for w_idx, word in enumerate(sentence):
        #         if final_mask[s_idx][w_idx] == 0.:
        #             break
        #         if word == 0:
        #             aspect_iter.append(0)
        #         elif word == 1:
        #             aspect_iter.append(1)
        #         elif word == 2:
        #             aspect_iter.append(2)
        #     aspect_out.append(aspect_iter)
        #
        # # logger.info(aspect_out)
        # # aspect_txt = open('data/{}/test/{}_pred_target.txt'.format(self.opt.target, 'DECNN'), 'w', encoding='utf-8')
        # #
        # #
        # # for sentence in aspect_out:
        # #     for idx, word in enumerate(sentence):
        # #         if idx == len(sentence) - 1:
        # #             aspect_txt.write(str(word) + '\n')
        # #         else:
        # #             aspect_txt.write(str(word) + ' ')

        return t_aspect_f1, t_opinion_f1, t_sentiment_acc, t_sentiment_f1, t_ABSA_f1, \
               t_loss.item(), t_aspect_loss.item(), t_opinion_loss.item(), t_sentiment_loss.item(), t_reg_loss.item()

    def run(self):
        # Loss and Optimizer
        # criterion = nn.CrossEntropyLoss()
        criterion = nn.NLLLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.lr_decay)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=len(self.testset), shuffle=False)
        # test_data_loader = DataLoader(dataset=self.testset, batch_size=(self.opt.batch_size), shuffle=False)
        dev_data_loader = DataLoader(dataset=self.valset, batch_size=len(self.valset), shuffle=False)
        # dev_data_loader = DataLoader(dataset=self.valset, batch_size=(self.opt.batch_size), shuffle=False)

        self._reset_params()
        pretrain_model_path = self._train(criterion, optimizer, train_data_loader, test_data_loader, dev_data_loader, is_training=True)

        # best_model_path = 'state_dict/{}/BEST-{}.pth'.format(self.opt.target, 'DECNN')
        # self.model.load_state_dict(torch.load(best_model_path))
        # self.model.eval()
        # test_aspect_f1, test_opinion_f1, test_sentiment_acc, test_sentiment_f1, test_ABSA_f1, \
        # test_loss, test_aspect_loss, test_opinion_loss, test_sentiment_loss, test_reg_loss = \
        #     self._evaluate_acc_f1(test_data_loader, 0, 0)
        # logger.info('>> test_f1: {:.4f}'.format(test_aspect_f1))
def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', default='restaurant', type=str, help='laptop device restaurant')
    parser.add_argument('--target', default='laptop', type=str, help='laptop device restaurant')
    parser.add_argument('--use_unlabel', default=0, type=int, help='use unlabel samples')
    parser.add_argument('--use_syntactic', default=0, type=int, help='use syntactic enhancement')
    parser.add_argument('--use_semantic', default=0, type=int, help='use semantic enhancement')
    parser.add_argument('--split', default=1, type=int, help='specify a split, 1, 2, 3')
    parser.add_argument('--local_bert_path', default=r'F:\NLP\BERT-BASE-UNCASED', type=str)
    parser.add_argument('--model_name', default='DECNN', type=str)
    parser.add_argument('--batch_size', default=8, type=int, help='number of example per batch')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=1e-5, type=float, help='learning rate decay')
    parser.add_argument('--num_epoch', default=100, type=int, help='training iteration')
    parser.add_argument('--emb_dim', default=100, type=int, help='dimension of word embedding')
    parser.add_argument('--dep_dim', default=50, type=int, help='dimension of word embedding')
    parser.add_argument('--pos_dim', default=50, type=int, help='dimension of word embedding')
    parser.add_argument('--hidden_dim', default=256, type=int, help='dimension of position embedding')
    parser.add_argument('--keep_prob', default=0.5, type=float, help='dropout keep prob')
    parser.add_argument('--l2_reg', default=1e-5, type=float, help='l2 regularization')
    parser.add_argument('--beta', default=0.1, type=float, help='interpolation')
    parser.add_argument('--kernel_size', default=5, type=int, help='kernel size')
    parser.add_argument('--hop_num', default=2, type=int, help='hop number')
    parser.add_argument('--class_num', default=3, type=int, help='class number')
    parser.add_argument('--cluster_num', default=200, type=int, help='class number')
    parser.add_argument('--cate_num', default=20, type=int, help='category number') # 20-RES16
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--dynamic_gate', default=1, type=int, help='use dynamic gate for harmonic prediction')
    parser.add_argument('--valset_num', default=0, type=int, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--reuse_embedding', default=1, type=int, help='reuse word embedding & id, True or False')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='uniform_', type=str)
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')
    parser.add_argument('--ablation', default='None', type=str, help='forward_lm, backward_lm, concat, gate, None')
    parser.add_argument('--topk', default=10, type=int, help='1~10')
    parser.add_argument('--name', default='BaseTagger', type=str)
    opt = parser.parse_args()
    start_time = time()
    print('Remote Check Success')
    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    model_classes = {
        'DECNN': DECNN
    }

    opt.dataset_file = {
            'train': './data/{}/train{}/'.format(opt.source, opt.split),
            'unlabel': './data/{}/train{}/'.format(opt.target, opt.split),
            'test': './data/{}/test{}/'.format(opt.target, opt.split)
    }
    input_colses = {
        'DECNN': ['sentence', 'mask', 'position', 'keep_prob']
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal,
        'orthogonal_': torch.nn.init.orthogonal_,
        'kaiming_uniform_':  torch.nn.init.kaiming_uniform_,
        'uniform_':  torch.nn.init.uniform_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    opt.max_sentence_len = 110

    if not os.path.exists('./log/{}-to-{}'.format(opt.source, opt.target)):
        os.makedirs('./log/{}-to-{}'.format(opt.source, opt.target))

    log_file = './log/{}-to-{}/{}-{}.log'.format(opt.source, opt.target, opt.name, strftime("%y%m%d-%H%M%S", localtime()))
    logger.addHandler(logging.FileHandler(log_file))
    logger.info('> log file: {}'.format(log_file))
    ins = Instructor(opt)
    ins.run()

    end_time = time()
    logger.info('Running Time: {:.0f}m {:.0f}s'.format((end_time-start_time) // 60, (end_time-start_time) % 60))

if __name__ == '__main__':
    main()
