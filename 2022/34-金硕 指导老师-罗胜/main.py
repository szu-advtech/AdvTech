# -*- coding: utf-8 -*-
import sys
import argparse

import torch
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import time

from data_loader import data_loader as data_loader
import options

import os

from learner.dnn import DNN
from learner.maml import MAML
from learner.metasense import MetaSense
from learner.pn import PN
from tensorboard_logger import Tensorboard

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)


def get_path(args):
    path = 'log/'

    # information about used data type
    path += args.dataset + '/'

    # information about used model type
    path += args.method + '/'

    # information about domain(condition) of training data
    if args.src == ['rest']:
        path += 'src_rest' + '/'
    elif args.src == ['all']:
        path += 'src_all' + '/'
    elif len(args.src) >= 1:
        path += 'src_' + '_'.join(args.src) + '/'

    if args.tgt:
        path += 'tgt_' + args.tgt + '/'

    path += args.log_suffix + '/'

    checkpoint_path = path + 'cp/'
    log_path = path + '/'
    result_path = path + '/'

    print('Path:{}'.format(path))
    return result_path, checkpoint_path, log_path


def main(args):
    ######################################################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print(device)

    ################### Hyper parameters #################
    if 'hhar' in args.dataset:
        if args.dataset == 'hhar_scaled':
            opt = options.HHAROpt
    elif 'dsa' in args.dataset:
        if args.dataset == 'dsa_scaled':
            opt = options.DSAOpt
    elif 'metasense_activity' in args.dataset:
        if args.dataset == 'metasense_activity_scaled':
            opt = options.MetaSense_ActivityOpt
    elif 'metasense_speech' in args.dataset:
        if args.dataset == 'metasense_speech_scaled':
            opt = options.MetaSense_SpeechOpt
            if args.method in ['Src', 'Tgt', 'Src_Tgt', 'TrC']: #prevent high fluctuation for non-meta learning method
                opt['learning_rate'] = opt['learning_rate']*0.1


    model = None
    if args.model == 'HHAR_model':
        import models.HHAR_model as model
    elif args.model == 'DSA_model':
        import models.DSA_model as model
    elif args.model == 'MetaSense_Activity_model':
        import models.MetaSense_Activity_model as model
    elif args.model == 'MetaSense_Activity_model_shallow':
        import models.MetaSense_Activity_model_shallow as model
    elif args.model == 'MetaSense_Activity_model_deep':
        import models.MetaSense_Activity_model_deep as model
    elif args.model == 'MetaSense_Speech_model':
        import models.MetaSense_Speech_model as model
    elif args.model == 'MetaSense_Speech_model_shallow':
        import models.MetaSense_Speech_model_shallow as model
    elif args.model == 'MetaSense_Speech_model_deep':
        import models.MetaSense_Speech_model_deep as model
    elif args.model == 'MetaSense_Activity_model_DeepConvLSTM':
        import models.MetaSense_Activity_model_DeepConvLSTM as model
    elif args.model == 'MetaSense_Activity_model_DeepConvLSTM_deepfc2_ecdf':
        import models.MetaSense_Activity_model_DeepConvLSTM_deepfc2_ecdf as model


    result_path, checkpoint_path, log_path = get_path(args)
    tensorboard = Tensorboard(log_path)

    ########## Dataset loading ############################
    source_data_loader = None
    target_data_loader = None
    learner = None

    if args.method == 'Src':

        print('##############Source Data Loading...##############')
        if args.motivation:
            source_data_loader = data_loader.domain_data_loader(args, args.src, opt['file_path'],
                                                                batch_size=opt['batch_size'],
                                                                valid_split=0.2,
                                                                test_split=0.2,
                                                                separate_domains=False, is_src=True)
        else:
            source_data_loader = data_loader.domain_data_loader(args, args.src, opt['file_path'],
                                                            batch_size=opt['batch_size'],
                                                            valid_split=0,
                                                            test_split=0,
                                                            separate_domains=False, is_src=True)

        print('##############Target Data Loading...##############')
        target_data_loader = data_loader.domain_data_loader(args, args.tgt, opt['file_path'],
                                                            batch_size=args.nshot,
                                                            valid_split=0.2,
                                                            test_split=0.2,
                                                            separate_domains=True, is_src=False)
        if args.src == ['all']:
            temp_source_data_loader, target_data_loader = data_loader.support_query_data_loader(
                target_data_loader=target_data_loader,
                query_split=0,
                batch_size=args.nshot,
                nshot=opt['batch_size'])
            # convert KSHOT dataset into normal dataset
            dataset = [x for x in temp_source_data_loader[0]['train'].dataset] # fix one dataset (or will shuffle for every call)
            features = torch.cat([x[0] for x in dataset], dim =0)
            class_labels = torch.cat([x[1] for x in dataset], dim=0)
            domain_labels = torch.cat([x[2] for x in dataset], dim=0)
            source_dataset_from_target = torch.utils.data.TensorDataset(features, class_labels, domain_labels)

            source_data_loader['train'] = data_loader.datasets_to_dataloader([source_dataset_from_target, source_data_loader['train'].dataset], batch_size=source_data_loader['train'].batch_size, concat=True, drop_last=True)
            source_data_loader['num_domains']+=1

        learner = DNN(args, opt, model, tensorboard=tensorboard, source_dataloader=source_data_loader, target_dataloader=target_data_loader[0])


    elif args.method == 'Tgt':

        print('##############Target Data Loading...##############')
        target_data_loader = data_loader.domain_data_loader(args, args.tgt, opt['file_path'],
                                                            batch_size=args.nshot,
                                                            valid_split=0.2,
                                                            test_split=0.2,
                                                            separate_domains=True, is_src=False)

        assert(args.src != ['all']) # Tgt does not have any source

        learner = DNN(args, opt, model, tensorboard=tensorboard, source_dataloader=source_data_loader,
                      target_dataloader=target_data_loader[
                          0])

    elif args.method == 'Src_Tgt':

        print('##############Source Data Loading...##############')
        source_data_loader = data_loader.domain_data_loader(args, args.src, opt['file_path'],
                                                            batch_size=opt['batch_size'],
                                                            valid_split=0,
                                                            test_split=0,
                                                            separate_domains=False, is_src=True)

        print('##############Target Data Loading...##############')
        target_data_loader = data_loader.domain_data_loader(args, args.tgt, opt['file_path'],
                                                            batch_size=args.nshot,
                                                            valid_split=0.2,
                                                            test_split=0.2,
                                                            separate_domains=True, is_src=False)
        if args.src == ['all']:
            temp_source_data_loader, target_data_loader = data_loader.support_query_data_loader(
                target_data_loader=target_data_loader,
                query_split=0,
                batch_size=args.nshot,
                nshot=opt['batch_size'])

            # convert KSHOT dataset into normal dataset
            dataset = [x for x in temp_source_data_loader[0]['train'].dataset] # fix one dataset (or will shuffle for every call)
            features = torch.cat([x[0] for x in dataset], dim =0)
            class_labels = torch.cat([x[1] for x in dataset], dim=0)
            domain_labels = torch.cat([x[2] for x in dataset], dim=0)
            source_dataset_from_target = torch.utils.data.TensorDataset(features, class_labels, domain_labels)

            source_data_loader['train'] = data_loader.datasets_to_dataloader([source_dataset_from_target, source_data_loader['train'].dataset], batch_size=source_data_loader['train'].batch_size, concat=True, drop_last=True)
            source_data_loader['num_domains']+=1

        learner = DNN(args, opt, model, tensorboard=tensorboard, source_dataloader=source_data_loader,
                      target_dataloader=target_data_loader[
                          0])

    elif args.method == 'TrC':

        print('##############Target Data Loading...##############')
        target_data_loader = data_loader.domain_data_loader(args, args.tgt, opt['file_path'],
                                                            batch_size=args.nshot,
                                                            valid_split=0.2,
                                                            test_split=0.2,
                                                            separate_domains=True, is_src=False)

        # for args.src==['all'] case, TrC will load the checkpoint of SrC trained with 'all' data. No further actions required.

        learner = DNN(args, opt, model, tensorboard=tensorboard, source_dataloader=source_data_loader,
                      target_dataloader=target_data_loader[
                          0])

    elif args.method == 'PN':

        print('##############Source Data Loading...##############')
        source_data_loader = data_loader.domain_data_loader(args, args.src, opt['file_path'],
                                                            batch_size=args.nshot,
                                                            valid_split=0,
                                                            test_split=0.5,
                                                            separate_domains=True, is_src=True)

        print('##############Target Data Loading...##############')
        target_data_loader = data_loader.domain_data_loader(args, args.tgt, opt['file_path'],
                                                            batch_size=10,
                                                            valid_split=0.2,
                                                            test_split=0.2,
                                                            separate_domains=True, is_src=False)
        if args.src == ['all']:
            temp_source_data_loader, target_data_loader = data_loader.support_query_data_loader(
                target_data_loader=target_data_loader,
                query_split=0.5,
                batch_size=10,
                nshot=args.nshot)
            source_data_loader.append(temp_source_data_loader[0])
            for dl in source_data_loader:
                dl['num_domains'] += 1

        learner = PN(args, opt, model, tensorboard=tensorboard, source_dataloaders=source_data_loader,
                     target_dataloader=target_data_loader[0])

    elif args.method == 'MAML':

        print('##############Source Data Loading...##############')
        source_data_loader = data_loader.domain_data_loader(args, args.src, opt['file_path'],
                                                            batch_size=args.nshot,
                                                            valid_split=0,
                                                            test_split=0.5,
                                                            separate_domains=True, is_src=True)

        print('##############Target Data Loading...##############')
        target_data_loader = data_loader.domain_data_loader(args, args.tgt, opt['file_path'],
                                                            batch_size=10,
                                                            valid_split=0.2,
                                                            test_split=0.2,
                                                            separate_domains=True, is_src=False)
        if args.src == ['all']:
            temp_source_data_loader, target_data_loader = data_loader.support_query_data_loader(
                target_data_loader=target_data_loader,
                query_split=0.5,
                batch_size=10,
                nshot=args.nshot)
            source_data_loader.append(temp_source_data_loader[0])
            for dl in source_data_loader:
                dl['num_domains'] += 1

        learner = MAML(args, opt, model, tensorboard=tensorboard, source_dataloaders=source_data_loader,
                       target_dataloader=target_data_loader[0], lr=args.lr)
    elif args.method == 'MetaSense':

        print('##############Source Data Loading...##############')
        source_data_loader = data_loader.domain_data_loader(args, args.src, opt['file_path'],
                                                            batch_size=args.nshot,
                                                            valid_split=0,
                                                            test_split=0.5,
                                                            separate_domains=True, is_src=True)

        print('##############Target Data Loading...##############')
        target_data_loader = data_loader.domain_data_loader(args, args.tgt, opt['file_path'],
                                                            batch_size=10,
                                                            valid_split=0.2,
                                                            test_split=0.2,
                                                            separate_domains=True, is_src=False)
        if args.src == ['all']:
            temp_source_data_loader, target_data_loader = data_loader.support_query_data_loader(
                target_data_loader=target_data_loader,
                query_split=0.5,
                batch_size=10,
                nshot=args.nshot)
            source_data_loader.append(temp_source_data_loader[0])
            for dl in source_data_loader:
                dl['num_domains'] += 1

        learner = MetaSense(args, opt, model, tensorboard=tensorboard, source_dataloaders=source_data_loader,
                                target_dataloader=target_data_loader[0], lr=args.lr,
                                is_shuffle_label=args.shuffle_label,
                                is_per_cond_task_only=args.per_cond_task_only)


    #################### Training #########################
    if args.train:

        since = time.time()

        start_epoch = 1
        best_acc = 0.0

        test_epoch = 5

        if args.load_checkpoint_path: # False if args.load_checkpoint_path==''
            resume = args.load_checkpoint_path + 'cp_best.pth.tar'
            if 'TrC' in args.method:
                learner.load_checkpoint(resume, True)
            else:
                learner.load_checkpoint(resume)

        for epoch in range(start_epoch, args.epoch + 1):
            if epoch == start_epoch:
                # make dir if doesn't exist
                if not os.path.exists(result_path):
                    oldumask = os.umask(0)
                    os.makedirs(result_path, 0o777)
                    os.umask(oldumask)
                if not os.path.exists(checkpoint_path):
                    oldumask = os.umask(0)
                    os.makedirs(checkpoint_path, 0o777)
                    os.umask(oldumask)
                for arg in vars(args):
                    tensorboard.log_text('args/' + arg, getattr(args, arg), 0)
                script = ' '.join(sys.argv[1:])
                tensorboard.log_text('args/script', script, 0)

            learner.train(epoch)


            if (epoch % test_epoch == 0):
                epoch_acc, epoch_loss = learner.validation(epoch)
                # keep best accuracy and model
                if (epoch_acc > best_acc):
                    best_acc = epoch_acc
                    learner.save_checkpoint(epoch=epoch, epoch_acc=epoch_acc, best_acc=best_acc,
                                            checkpoint_path=checkpoint_path + 'cp_best.pth.tar')

        time_elapsed = time.time() - since

        print('Training complete time: {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

    ############### Test model #########################
    if args.train: # if it's train, load the new model in the new path
        resume = checkpoint_path + 'cp_best.pth.tar'
    else:
        resume = args.load_checkpoint_path + 'cp_best.pth.tar'

    checkpoint = learner.load_checkpoint(resume)
    if checkpoint != None:
        best_valid_acc = checkpoint['best_acc']
    else:
        best_valid_acc = 0
    class_accuracy_of_test_data = learner.test(args.epoch)

    if isinstance(class_accuracy_of_test_data, (list,)): # for MAML and MetaSense, results of multiple shots (1..10)

        result_text = 'Validation best_acc:\t{:f}\n'.format(
            best_valid_acc)
        for i,acc in enumerate(class_accuracy_of_test_data):
            result_text += 'Test Class Accuracy {:d}shot:\t{:f}\n'.format(i+1, acc)
    else:
        # Record args, script, result in Tensorboard log
        result_text = 'Validation best_acc:\t{:f}\nTest Class Accuracy:\t{:f}'.format(
            best_valid_acc, class_accuracy_of_test_data)

    tensorboard.log_text('result/', result_text, 0)
    f = open(result_path + 'result.txt', 'w')
    f.write(result_text)
    f.close()

    tensorboard.close()


def parse_arguments(argv):
    """Command line parse."""

    # Note that 'type=bool' args should be False in default. Any string argument is recognized as "True". Do not give "--bool_arg 0"

    parser = argparse.ArgumentParser()

    ###MANDATORY###

    parser.add_argument('--train', type=bool, default=False,
                        help='do train? or only test')
    parser.add_argument('--dataset', type=str, default='',
                        help='Dataset to be used, in [metasense_activity_scaled, metasense_speech_scaled, dsa_scaled, hhar_scaled]')

    parser.add_argument('--model', type=str, default='HHAR_model',
                        help='Which model to use')

    parser.add_argument('--method', type=str, default='',
                        help='Src'
                             'Tgt'
                             'Src_Tgt'
                             'TrC'
                             'MAML'
                             'PN'
                             'MetaSense')

    parser.add_argument('--src', nargs='*', required=True, default=None,
                        help='Specify source dataset (labeled)'
                             'rest'
                             'all (used for metasense extension)'
                             'nexus5x.neuxs5x_1.a')

    parser.add_argument('--tgt', type=str, default='',
                        help='specific test domain. currently support only one domain')

    ###Optional###
    parser.add_argument('--lr', type=float, default=0.0005,
                        help='learning rate')

    parser.add_argument('--epoch', type=int, default=200,
                        help='How many epochs do you want to use for train')
    parser.add_argument('--load_checkpoint_path', type=str, default='',
                        help='Load checkpoint and train from checkpoint in path?')
    parser.add_argument('--dynamic_constant', type=bool, default=False,
                        help='Set a dynamic constant multiplied to the gradient reversal layer')

    parser.add_argument('--train_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for train')
    parser.add_argument('--valid_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for valid')
    parser.add_argument('--test_max_rows', type=int, default=np.inf,
                        help='How many data do you want to use for test')
    parser.add_argument('--nshot', type=int, default=1,
                        help='How many shot do you want use for meta-learning?')
    parser.add_argument('--log_suffix', type=str, default='',
                        help='Suffix of log file path')

    ### MetaSense-specific ###
    parser.add_argument('--shuffle_label', type=bool, default=False,
                        help='Shuffle labels for maml')
    parser.add_argument('--per_cond_task_only', type=bool, default=False,
                        help='Set a dynamic constant multiplied to the gradient reversal layer')
    parser.add_argument('--motivation', type=bool, default=False,
                        help='motivation section for metasense_speech')
    parser.add_argument('--num_source', type=int, default=100,
                        help='number of available sources')
    parser.add_argument('--num_bin', type=int, default=50,
                        help='number of bins in ecdf')

    return parser.parse_args()

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
