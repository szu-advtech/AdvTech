# -*- coding: utf-8 -*-
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import os
import logging
import argparse
import math
import sys
from time import strftime, localtime, time
import random
import numpy

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
from pytorch_transformers import BertModel
from torch.utils.data import DataLoader, random_split

from utils import ABSADataset
from evaluation import *
from models.decnn import DECNN

print('Model is DECNN')

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


class Instructor:
    def __init__(self, opt):
        self.opt = opt

        'word index => word2id.txt'
        'word embedding => word_embedding.npy'

        print('Reuse Word Dictionary & Embedding')
        with open('./data/word2id.txt', 'r', encoding='utf-8') as f:
            word_dict = eval(f.read())
        w2v_global = np.load('./data/glove_embedding.npy')
        if opt.dataset in ['res14', 'res15', 'res16']:
            w2v_domain = np.load('./data/restaurant_embedding.npy')
        elif opt.dataset in ['lap14']:
            w2v_domain = np.load('./data/laptop_embedding.npy')
        else:
            print('Error in Dataset!')
            raise ValueError

        self.model = opt.model_class(w2v_global, w2v_domain, opt).to(opt.device)

        self.trainset = ABSADataset(opt.dataset_file['train'], word_dict, opt)
        self.testset = ABSADataset(opt.dataset_file['test'],word_dict, opt)

        if opt.valset_num > 0:
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
        for child in self.model.children():
            if type(child) != BertModel:  # skip bert params
                for p in child.parameters():
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
            aspect_loss_total, opinion_loss_total, sentiment_loss_total, reg_loss_total = 0., 0., 0., 0.
            # switch model to training mode
            self.model.train()
            for i_batch, sample_batched in enumerate(train_data_loader):
                global_step += 1
                # clear gradient accumulators
                optimizer.zero_grad()
                if self.opt.lm in ['internal', 'external']:
                    inputs = [sample_batched[col].to(self.opt.device) for col in ['x', 'mask',
                                                                                  'fw_lmwords', 'fw_lmprobs',
                                                                                  'bw_lmwords','bw_lmprobs']]
                elif self.opt.lm in ['bert_base', 'bert_pt']:
                    inputs = [sample_batched[col].to(self.opt.device) for col in ['x', 'mask', 'lmwords', 'lmprobs']]
                else:
                    inputs = [sample_batched[col].to(self.opt.device) for col in ['x', 'mask']]

                aspect_y = sample_batched['aspect_y'].to(self.opt.device)
                outputs = torch.clamp(self.model(inputs, epoch, tau_now, is_training=is_training, train_y=aspect_y), 1e-5, 1.)

                length = torch.sum(inputs[1])
                # loss = criterion(outputs, aspect_y)
                aspect_loss = torch.sum(-1 * torch.log(outputs) * aspect_y.float()) / (length + 1e-6)
                opinion_loss = torch.tensor(0.)
                sentiment_loss = torch.tensor(0.)
                reg_loss = torch.tensor(0.)
                loss = aspect_loss + opinion_loss + sentiment_loss + self.opt.l2_reg * reg_loss
                loss.backward()
                optimizer.step()
                n_total += length
                loss_total += loss.item() * length
                aspect_loss_total += aspect_loss.item() * length
                opinion_loss_total += opinion_loss.item() * length
                sentiment_loss_total += sentiment_loss.item() * length
                reg_loss_total += reg_loss.item() * length
            train_loss = loss_total / n_total
            train_aspect_loss = aspect_loss_total / n_total
            train_opinion_loss = opinion_loss_total / n_total
            train_sentiment_loss = sentiment_loss_total / n_total
            train_reg_loss = reg_loss_total / n_total

            'DEV'
            dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1, \
            dev_loss, dev_aspect_loss, dev_opinion_loss, dev_sentiment_loss, dev_reg_loss = \
            self._evaluate_acc_f1(dev_data_loader, epoch, tau_now)
            dev_metric = dev_aspect_f1
            if epoch < 100:
                dev_metric_list.append(0.)
                dev_loss_list.append(1000.)
            else:
                dev_metric_list.append(dev_metric)
                dev_loss_list.append(dev_loss)

            save_indicator = 0
            if (dev_metric > max_dev_metric or dev_loss < min_dev_loss) and epoch >= 100:
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
            logger.info('Train: final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                        format(train_loss, train_aspect_loss, train_opinion_loss, train_sentiment_loss, train_reg_loss, global_step))
            logger.info('Dev:   final loss={:.6f}, aspect loss={:.6f}, opinion loss={:.6f}, sentiment loss={:.6f}, reg loss={:.6f}, step={}'.
                        format(dev_loss, dev_aspect_loss, dev_opinion_loss, dev_sentiment_loss, dev_reg_loss, global_step))
            logger.info('Dev:   aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                        .format(dev_aspect_f1, dev_opinion_f1, dev_sentiment_acc, dev_sentiment_f1, dev_ABSA_f1))
            logger.info('Test:  aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                        .format(test_aspect_f1, test_opinion_f1, test_sentiment_acc, test_sentiment_f1, test_ABSA_f1))
            logger.info('Current Max Metrics Index : {} Current Min Loss Index : {} {} Tau : {:.2f}'
                        .format(dev_metric_list.index(max(dev_metric_list)), dev_loss_list.index(min(dev_loss_list)), epoch_time, tau_now))

            'SAVE CheckPoints'
            # if not os.path.exists('state_dict/{}'.format(self.opt.dataset)):
            #     os.mkdir('state_dict/{}'.format(self.opt.dataset))
            # if save_indicator == 1:
            #     path = 'state_dict/{}/BEST-{}-testF1({:.2f}).pth'.format(self.opt.dataset, self.opt.lm, test_aspect_f1 * 100)
            #     torch.save(self.model.state_dict(), path)
            #     logger.info('>> Checkpoint Saved: {}'.format(path))



        'SUMMARY'
        logger.info('\n{:-^80}'.format('Mission Complete'))
        max_dev_index = dev_metric_list.index(max(dev_metric_list))
        logger.info('Dev Max Metrics Index: {}'.format(max_dev_index))
        logger.info('aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                    .format(aspect_f1_list[max_dev_index], opinion_f1_list[max_dev_index], sentiment_acc_list[max_dev_index],
                            sentiment_f1_list[max_dev_index], ABSA_f1_list[max_dev_index]))

        min_dev_index = dev_loss_list.index(min(dev_loss_list))
        logger.info('Dev Min Loss Index: {}'.format(min_dev_index))
        logger.info('aspect f1={:.4f}, opinion f1={:.4f}, sentiment acc=={:.4f}, sentiment f1=={:.4f}, ABSA f1=={:.4f},'
                    .format(aspect_f1_list[min_dev_index], opinion_f1_list[min_dev_index], sentiment_acc_list[min_dev_index],
                            sentiment_f1_list[min_dev_index], ABSA_f1_list[min_dev_index]))

        return path

    def _evaluate_acc_f1(self, data_loader, epoch, tau_now):
        n_correct, n_total, loss_total = 0, 0, 0
        aspect_loss_total, opinion_loss_total, sentiment_loss_total, reg_loss_total = 0., 0., 0., 0.
        t_aspect_y_all, t_outputs_all, t_mask_all = list(), list(), list()
        # switch model to evaluation mode
        self.model.eval()
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(data_loader):
                if self.opt.lm in ['internal', 'external']:
                    inputs = [t_sample_batched[col].to(self.opt.device) for col in ['x', 'mask',
                                                                                  'fw_lmwords', 'fw_lmprobs',
                                                                                  'bw_lmwords','bw_lmprobs']]
                elif self.opt.lm in ['bert_base', 'bert_pt']:
                    inputs = [t_sample_batched[col].to(self.opt.device) for col in ['x', 'mask', 'lmwords', 'lmprobs']]
                else:
                    inputs = [t_sample_batched[col].to(self.opt.device) for col in ['x', 'mask']]
                t_outputs = torch.clamp(self.model(inputs, epoch, tau_now), 1e-5, 1.)
                t_aspect_y = t_sample_batched['aspect_y'].to(self.opt.device)

                length = torch.sum(inputs[1])

                aspect_loss = torch.sum(-1 * torch.log(t_outputs) * t_aspect_y.float()) / length
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
                t_outputs_all.extend(t_outputs.cpu().tolist())
                t_mask_all.extend(inputs[1].cpu().tolist())
            t_loss = loss_total / n_total
            t_aspect_loss = aspect_loss_total / n_total
            t_opinion_loss = opinion_loss_total / n_total
            t_sentiment_loss = sentiment_loss_total / n_total
            t_reg_loss = reg_loss_total / n_total
        t_aspect_f1, t_opinion_f1, t_sentiment_acc, t_sentiment_f1, t_ABSA_f1 = get_metric(t_aspect_y_all, t_outputs_all,
                                           np.zeros_like(t_aspect_y_all),  np.zeros_like(t_aspect_y_all),
                                           np.zeros_like(t_aspect_y_all),  np.zeros_like(t_aspect_y_all),
                                           t_mask_all, 1)

        return t_aspect_f1, t_opinion_f1, t_sentiment_acc, t_sentiment_f1, t_ABSA_f1, \
               t_loss.item(), t_aspect_loss.item(), t_opinion_loss.item(), t_sentiment_loss.item(), t_reg_loss.item()

    def run(self):
        # Loss and Optimizer
        criterion = nn.NLLLoss()
        _params = filter(lambda p: p.requires_grad, self.model.parameters())
        optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.lr_decay)

        train_data_loader = DataLoader(dataset=self.trainset, batch_size=self.opt.batch_size, shuffle=True)
        test_data_loader = DataLoader(dataset=self.testset, batch_size=len(self.testset), shuffle=False)
        dev_data_loader = DataLoader(dataset=self.valset, batch_size=len(self.valset), shuffle=False)

        self._reset_params()
        pretrain_model_path = self._train(criterion, optimizer, train_data_loader, test_data_loader, dev_data_loader, is_training=True)

        # self.model.load_state_dict(torch.load(pretrain_model_path))
        # best_model_path = self._train(criterion, optimizer, train_data_loader, test_data_loader, dev_data_loader, pre_training=False)
        # self.model.load_state_dict(torch.load(best_model_path))
        # self.model.eval()
        # test_f1 = self._evaluate_acc_f1(test_data_loader)
        # logger.info('>> test_f1: {:.4f}'.format(test_f1))


def main():
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='res16', type=str, help='res14 lap14 res15 res16')
    parser.add_argument('--model_name', default='DECNN', type=str)
    parser.add_argument('--batch_size', default=8, type=int, help='number of example per batch')
    parser.add_argument('--learning_rate', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=1e-5, type=float, help='learning rate decay')
    parser.add_argument('--num_epoch', default=200, type=int, help='training iteration')
    parser.add_argument('--emb_dim', default=400, type=int, help='dimension of word embedding')
    parser.add_argument('--hidden_dim', default=400, type=int, help='dimension of position embedding')
    parser.add_argument('--keep_prob', default=0.5, type=float, help='dropout keep prob')
    parser.add_argument('--l2_reg', default=1e-5, type=float, help='l2 regularization')
    parser.add_argument('--lm', default='None', type=str, help='language models: internal, external, bert_base, bert_pt')
    parser.add_argument('--class_num', default=3, type=int, help='class number')
    parser.add_argument('--seed', default=123, type=int, help='random seed')
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--valset_num', default=150, type=int, help='set ratio between 0 and 1 for validation support')
    parser.add_argument('--reuse_embedding', default=1, type=int, help='reuse word embedding & id, True or False')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='uniform_', type=str)
    parser.add_argument('--device', default='cuda:0', type=str, help='e.g. cuda:0')

    opt = parser.parse_args()
    start_time = time()
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
    dataset_files = {
        'res14': {
            'train': './data/res14/train/',
            'test': './data/res14/test/'
        },
        'res15': {
            'train': './data/res15/train/',
            'test': './data/res15/test/'
        },
        'res16': {
            'train': './data/res16/train/',
            'test': './data/res16/test/'
        },
        'lap14': {
            'train': './data/lap14/train/',
            'test': './data/lap14/test/'
        }
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
    topk_dict = {'lap14': 10, 'res14': 7,'res15': 10, 'res16': 7}
    opt.topk = topk_dict[opt.dataset]

    max_length_dict = {'lap14': 85, 'res14': 80,'res15': 70, 'res16': 75}
    opt.max_sentence_len = max_length_dict[opt.dataset]


    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.dataset_path = './data/{}/'.format(opt.dataset)
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if not os.path.exists('./log/{}'.format(opt.dataset)):
        os.makedirs('./log/{}'.format(opt.dataset))
    log_file = './log/{}/{}-{}-{}.log'.format(opt.dataset, opt.lm, opt.dataset, strftime("%y%m%d-%H%M%S", localtime()))
    logger.addHandler(logging.FileHandler(log_file))
    logger.info('> log file: {}'.format(log_file))
    ins = Instructor(opt)
    ins.run()

    end_time = time()
    logger.info('Running Time: {:.0f}m {:.0f}s'.format((end_time-start_time) // 60, (end_time-start_time) % 60))

if __name__ == '__main__':
    main()
