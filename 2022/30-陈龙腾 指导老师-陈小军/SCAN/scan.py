"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import argparse
import os
import torch
import sys

from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate
from utils.evaluate_utils import get_predictions, scan_evaluate, cc_evaluate, hungarian_evaluate
from utils.train_utils import mix_train, scan_train
from utils.utils import Logger, get_knn_indices
from torch.utils import data
import numpy as np
import time
import pandas as pd
import copy


FLAGS = argparse.ArgumentParser(description='SCAN Loss')
FLAGS.add_argument('--config_env', default="configs/env.yml", help='Location of path config file')
FLAGS.add_argument('--config_exp', default="configs/scan/scan_stl10.yml", help='Location of experiments config file')
FLAGS.add_argument('--version', type=str, default='stl10_scan-draw', help='Record the version of this times')
FLAGS.add_argument('--gpu', type=str, default='0,1')
FLAGS.add_argument('--t', type=float, default=8.0)
FLAGS.add_argument('--u', type=float, default=1)
FLAGS.add_argument('--topk', type=int, default=20)
FLAGS.add_argument('--R', type=int, default=0.5)
FLAGS.add_argument('--minibatch', type=int, default=64)
FLAGS.add_argument('--wo1', type=int, default=1)
FLAGS.add_argument('--wo2', type=int, default=0)
FLAGS.add_argument('--wo3', type=int, default=1)
FLAGS.add_argument('--checkpoint', type=str, default='checkpoint.pth.tar')
args = FLAGS.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu


def main():
    p = create_config(args.config_env, args.config_exp, args.topk, args.checkpoint, '')
    print(colored(p, 'red'))

    # log
    logfile_dir = os.path.join(os.getcwd(), 'logs/')
    logfile_name = logfile_dir + args.version + '.log'
    sys.stdout = Logger(filename=logfile_name, stream=sys.stdout)
    localtime = time.asctime(time.localtime(time.time()))
    print('\n--------------------------------------------------------------\n')
    print("The current time:", localtime)
    print('dataset_name: ', p['train_db_name'])

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    train_dataset, indices = get_train_dataset(p, val_transformations, flag=0, split='train', to_neighbors_dataset=True) # 不加增强
    loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler_m, num_workers=p['num_workers'], pin_memory=True)

    val_dataset = get_val_dataset(p, args, val_transformations, to_neighbors_dataset=True)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))

    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p, p['pretext_model'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # for i, (input) in enumerate(train_dataloader):
    #     # measure data loading time
    #     print(input.size())



    # by reading the acc, check whether the model parameters are loaded correctly
    # base_dataset = data.ConcatDataset([get_train_dataset(p, val_transformations, split='train', to_neighbors_dataset = True), get_val_dataset(p, val_transformations, to_neighbors_dataset = True)])
    # base_dataset = get_train_dataset(p, val_transformations,split='train', to_neighbors_dataset = True)
    # base_dataloader = get_val_dataloader(p, base_dataset)
    # from utils.utils import get_features_eval
    # features, targets = get_features_eval(base_dataloader, model, forward_pass='backbone')
    #
    # for i in range(10):
    #     from utils.evaluate_utils import kmeans
    #     kmeans(features, targets)



    base_dataset = get_train_dataset(p, val_transformations, flag=1, split='train', to_neighbors_dataset = False)
    dataloader = get_val_dataloader(p, base_dataset)
    from utils.utils import get_features_eval
    features, targets = get_features_eval(dataloader, model, forward_pass='backbone')
    from sklearn.cluster import KMeans

    # 画图
    print("draw the clustering result")
    from utils.evaluate_utils import kmeans
    for i in range(1):
        kmeans(features, targets)

    # get k-means centers
    kmeans1 = KMeans(p['num_classes'], n_init=20)
    predicted = kmeans1.fit_predict(features.cpu().numpy())
    predicted = torch.from_numpy(predicted)
    predicted = predicted.cuda(targets.device)


    num_elems = len(base_dataset)
    from utils.evaluate_utils import _hungarian_match
    match = _hungarian_match(predicted, targets, preds_k=p['num_classes'], targets_k=p['num_classes'])
    reordered_preds = torch.zeros(num_elems, dtype=predicted.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predicted == int(pred_i)] = int(target_i)
    from sklearn import metrics
    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predicted.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predicted.cpu().numpy())
    print({'ACC': acc, 'ARI': ari, 'NMI': nmi})



    # get knn indices
    # base_dataset = get_train_dataset(p, val_transformations, split='train', to_neighbors_dataset = True)
    # base_dataloader = get_val_dataloader(p, base_dataset)
    # state = torch.load(p['scan_model'], map_location='cpu')
    # print(p['scan_model'])
    # model.module.load_state_dict(state['model'], strict=True)
    # indices, acc = get_knn_indices(base_dataloader, model, topk=p['num_neighbors'])
    # np.save(p['after_top{}_neighbors_train_path'.format(p['num_neighbors'])], indices)
    # print(acc)

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)

    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: SCAN will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    from losses.losses import SCANLoss
    criterion = SCANLoss(args.wo1, args.wo2, args.wo3, args.t, args.u, p['num_classes'], **p['criterion_kwargs'])
    criterion = criterion.cuda()
    print(criterion)



    # Checkpoint
    if False and os.path.exists(p['scan_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['scan_checkpoint']), 'blue'))
        checkpoint = torch.load(p['scan_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']
        best_acc = 0
        best_clustering_stats = None

    else:
        print(colored('No checkpoint file at {}'.format(p['scan_checkpoint']), 'blue'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = 0
        best_acc = 0
        best_clustering_stats = None
        best_model = None

    # Main loop
    print(colored('Starting main loop', 'blue'))

    # 记录运行的开始时间
    torch.cuda.synchronize()
    start = time.time()

    results = [[] for i in range(6)]
    epoch_num = p['epochs']
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))

        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))

        # Train
        print('Train ...')
        final_loss = scan_train(train_dataloader, model, criterion, optimizer, epoch, epoch_num,
                                p['update_cluster_head_only'])


        # Evaluate
        print('Make prediction on validation set ...')
        predictions = get_predictions(p, val_dataloader, model)

        print('Evaluate with hungarian matching algorithm ...')
        lowest_loss_head = 0
        clustering_stats = hungarian_evaluate(lowest_loss_head, predictions, compute_confusion_matrix=False)
        print(clustering_stats)


        for i in range(3):
            results[i].append(final_loss[i])
        results[3].append(round(clustering_stats['ACC'],3))
        results[4].append(round(clustering_stats['ARI'],3))
        results[5].append(round(clustering_stats['NMI'],3))

        if clustering_stats['ACC'] > best_acc:
            best_acc = clustering_stats['ACC']
            best_clustering_stats = clustering_stats
            best_model = copy.deepcopy(model)


        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
                    'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head},
                     p['scan_checkpoint'])

    #记录运行结束的时间
    torch.cuda.synchronize()
    end = time.time()
    t = end - start

    print('=============================================================================================')
    print('parameter:',  '\tt:', args.t, '\tu:', args.u)
    print('best_clustering_stats:')
    print(best_clustering_stats)
    print('scan ' + p['train_db_name'] + ' The training time: {:.0f}min {:.0f}sec'.format(t // 60, t % 60))
    print('=============================================================================================')

    torch.save({'model': best_model.module.state_dict(), 'head': best_loss_head},
               os.path.join(p['scan_model'], 'model'+str(round(best_clustering_stats['ACC']*100, 2))+'.pth.tar'))


    # write the results to the excel file

    para = p['setup'] + p['train_db_name'] + ' u=' + str(args.u) + ' t=' + str(args.t) + ' k=' + str(args.topk)   # the parameter name and value
    print(para)
    print(args.wo1, args.wo2, args.wo3)
    index = np.arange(1, epoch_num+1)
    res = {para: index, "total_loss": results[0], "consistency_loss": results[1], "contrastive_loss": results[2],"ACC": results[3], "ARI": results[4], "NMI": results[5]}
    from pandas.core.frame import DataFrame
    res = DataFrame(res)

    file_name = '../results/stl10_scan_loss_compare.csv'
    res.to_csv(file_name, index=False, mode='a+', encoding='utf-8')

    # from openpyxl import load_workbook
    # file_name = '../results/results_para_2.xlsx'
    # res_old = pd.read_excel(file_name, sheet_name='scan')
    # with pd.ExcelWriter(file_name, mode='a', engine='openpyxl') as writer:
    #     book = load_workbook(file_name)
    #     writer.book = book
    #     writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
    #     row = res_old.shape[0]
    #     res.to_excel(writer, sheet_name='scan', startrow=row+2, index=False)


if __name__ == "__main__":
    main()
