"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
from datetime import datetime

import torch

from utils.utils import Logger, get_knn
from utils.config import create_config
from utils.common_config import get_train_dataset, get_train_transformations, get_weak_transformations,\
                                get_val_dataset, get_val_transformations,\
                                get_train_dataloader, get_val_dataloader,\
                                get_optimizer, get_model, adjust_learning_rate, get_train_dataloader_nosuf
from utils.ema import EMA
from utils.evaluate_utils import get_predictions, get_predictions_propa, hungarian_evaluate
from utils.train_utils import selflabel_train
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

import sys
import time
import numpy as np
import copy
import utils.helpers as helpers
from utils.utils import get_pseudo_labels
from utils.evaluate_utils import get_predictions_our_example, get_predictions_scan_example
from utils.helpers import create_data_loaders_simple

# Parser
parser = argparse.ArgumentParser(description='Self-labeling')
parser.add_argument('--config_env', default='configs/env.yml',
                    help='Config file for the environment')
parser.add_argument('--config_exp', default='configs/selflabel/selflabel_stl10.yml',
                    help='Config file for the experiment')
parser.add_argument('--version', type=str, default='stl10_0.9_0.01_v2', help='Record the version of this times') # default='stl10_0.95_0.03_epo100'

parser.add_argument('--checkpoint', type=str, default='version.pth.tar')
parser.add_argument('--P', type=float, default=0.8)
parser.add_argument('--R1',type=float, default=0.8)
parser.add_argument('--R2', type=float, default=0.99)
parser.add_argument('--R', type=int, default=0.5)
parser.add_argument('--alpha1', type=list, default=[1, 1, 1, 1], help="alpha for dirichelet") # 用于采样

parser.add_argument('--batch_size', type=int, default=256) # default=160,config文件中间有定义了
parser.add_argument('--minibatch', type=int, default=64)
parser.add_argument('--labeled_batch_size', type=int, default=128)
parser.add_argument('--mix_batch_size', type=int, default=16)
parser.add_argument('--Propa_num', type=int, default=5, help="Number of label propagation")
parser.add_argument('--knum', type=int, default=9, help="The fixed nearest neighbor visits the nearest neighbor region")
parser.add_argument('--mix_num', type=int, default=1, help="Number of sample mixes") # 需要减去自身
parser.add_argument('--sigma', type=float, default=1.5, help='the sigma of affinity matrix')
parser.add_argument('--ct2', type=float, default=0.9) # 0.995
parser.add_argument('--num', type=float, default=0.01, help='the sigma of affinity matrix') # 0.8
parser.add_argument('--topk_lp', type=int, default=50) # 标签传播k为50
parser.add_argument('--epochs', type=int, default=50) # default=810
parser.add_argument('--best_scan_model', type=str, default="best_model8009.pth.tar") # "model54.53.pth.tar"


parser.add_argument('--gpu', type=str, default='0, 1, 2, 3') # ‘0，1，2，3’
parser.add_argument('--ct1', type=float, default=0.6)
parser.add_argument('--eta', type=float, default=0.6) # the proportion of hardening
parser.add_argument('--topk', type=int, default=20) # 20个最近邻
parser.add_argument('--loss', type=str, default="our")
parser.add_argument('--aug_num', type=int, default=3)
parser.add_argument('--alpha', type=float, default=1.0,
                    help='mixup alpha for beta dis')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='max learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--nesterov', default=True, type=bool,
                    help='use nesterov momentum', metavar='BOOL')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0)


args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def main():
    # Retrieve config file
    p = create_config(args.config_env, args.config_exp, args.topk, args.checkpoint, args.best_scan_model)
    print(colored(p, 'red'))
    print(p['top{}_neighbors_train_path'.format(p['num_neighbors'])])

    # log
    logfile_dir = os.path.join(os.getcwd(), p['DIR']+'logs/')
    logfile_name = logfile_dir + args.version + '.log'
    sys.stdout = Logger(filename=logfile_name, stream=sys.stdout)
    localtime = time.asctime(time.localtime(time.time()))
    print('\n--------------------------------------------------------------\n')
    print("The current time:", localtime)
    para = p['setup'] + ' ' + p['train_db_name'] + '  ct2:' + str(args.ct2) + ' topk_lp:' + str(args.topk_lp) + ' gamma:' + str(args.gamma) +'\n'
    print(para)

    # Get model
    print(colored('Retrieve model', 'blue'))
    model = get_model(p, p['best_scan_model'])
    print(p['best_scan_model'])
    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # CUDNN
    print(colored('Set CuDNN benchmark', 'blue'))
    torch.backends.cudnn.benchmark = True

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=0.9, weight_decay=args.weight_decay, nesterov=args.nesterov)
    # Dataset
    print(colored('Retrieve dataset', 'blue'))

    # Transforms
    strong_transforms = get_train_transformations(p)
    weak_transforms = get_weak_transformations(p)
    val_transforms = get_val_transformations(p)
    train_dataset, indices = get_train_dataset(p, val_transforms, 0, split='train', to_neighbors_dataset=True) # 多返回一个近邻

    # train_dataset = get_train_dataset(p, val_transforms, to_neighbors_dataset=True)
    train_dataloader = get_train_dataloader(p, train_dataset) # the same between get_train_dataloader and get_train_dataloader_nosuf
    train_dataloader_nosuf = get_train_dataloader_nosuf(p, get_train_dataset(p, val_transforms, 1, split='train'))  # 不加增强
    val_dataset = get_val_dataset(p, args, val_transforms)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print(colored('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)), 'yellow'))



    # # 画图
    # base_dataset = get_train_dataset(p, val_transforms, split='train', to_neighbors_dataset=True)
    # dataloader = get_val_dataloader(p, base_dataset)
    # from utils.utils import get_features_eval
    # features, targets = get_features_eval(dataloader, model, forward_pass='backbone')
    # print("draw the clustering result")
    # from utils.evaluate_utils import kmeans
    # for i in range(1):
    #     kmeans(features, targets)

    # get the pseudo labels by ct2
    # predictions, sum = get_predictions_our_example(p, train_dataloader, model, args.ct2)
    predictions, sum = get_predictions_scan_example(p, train_dataloader, model, args.ct2)
    clustering_stats = hungarian_evaluate(0, predictions, compute_confusion_matrix=False)
    print(clustering_stats)
    print("the sum of the pseudo labels:", sum) # 实际使用的会删除，因为需要一样多的伪标签
    print("ct:", args.ct2, "  proportion:", sum/len(train_dataset)*100)

    # 获得的伪标签和索引用于构造 强伪标签数据集，所以那个标签传播需要在前面完成
    plabels, pl_indices = get_pseudo_labels(predictions, clustering_stats['hungarian_match']) # 返回的是带顺序的前topk伪标签，每个类的伪标签数量一致。

    #!!! get the pseudo labels by Label propagation
    # feats, tar = helpers.extract_features_simp(train_dataloader_nosuf, model, args, forward_pass='backbone')
    # # plabels, pl_indices = get_predictions_propa(args, p, train_dataloader, model) # 得到强标签，失败，因为太大了
    # D, I = get_knn(args.topk_lp, feats)
    # probs_iter, p_labels = Rinking(p, D, I, tar, args.topk_lp, iter=100)
    

    train_dataset2 = get_train_dataset(p, {'standard': val_transforms, 'weak': weak_transforms, 'augment': strong_transforms}, 1,
                                        split='train', debug=False) # 加增强了

    # Checkpoint
    if False and os.path.exists(p['selflabel_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['selflabel_checkpoint']), 'blue'))
        checkpoint = torch.load(p['selflabel_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_acc']
        dataset = checkpoint['dataset']

        train_loader, train_loader_noshuff, train_loader_l, train_loader_u, dataset = create_data_loaders_simple(
            args, p, train_dataset2, plabels, pl_indices, indices, data=dataset, checkpoint=True)

    else:
        print(colored('No checkpoint file at {}'.format(p['selflabel_checkpoint']), 'blue'))
        start_epoch = 0
        best_acc = 0
        # train_loader, train_loader_noshuff, train_loader_l, train_loader_u, train_loader_b, dataset = create_data_loaders_simple(args, p, train_dataset2, plabels, pl_indices, indices)
        train_loader, train_loader_noshuff, train_loader_l, train_loader_u, dataset = create_data_loaders_simple(
            args, p, train_dataset2, plabels, pl_indices, indices) # 返回dataset

    best_clustering_stats = None
    best_model = None

    # for batch in train_loader_l:
    #     batch = batch.to(args.device)

    #### Information store in epoch results and then saved to file
    global_step = 0
    args.lr_rampdown_epochs = args.epochs + 10    ### what is the meaning?
    args.progress = True
    args.device = torch.device('cuda')

    # Evaluate
    # print('Evaluate ...')
    # predictions = get_predictions(p, val_dataloader, model)
    # clustering_stats = hungarian_evaluate(0, predictions,
    #                                       class_names=val_dataset.classes,
    #                                       compute_confusion_matrix=True,
    #                                       confusion_matrix_file=os.path.join(p['selflabel_dir'],
    #                                                                          'confusion_matrix.png'))
    # print(clustering_stats)

    now = datetime.now()
    path_suffix = now.strftime('%m-%d_%H:%M:%S')
    tensorPath = os.path.join('output/baseline/selflabel', p['train_db_name'], args.version, path_suffix)
    writer = SummaryWriter(log_dir=tensorPath)
    results = [[] for i in range(10)]
    epoch_num = args.epochs
    print('Training without Smooth distribution alignment, with weak transforms')


    for epoch in range(start_epoch, args.epochs):
        #### Extract features and run label prop on graph laplacian
        print('Epoch {}/{}'.format(epoch+1, args.epochs))
        dataset.feat_mode = True
        feats, target_all = helpers.extract_features_simp(train_dataloader_nosuf, model, args, forward_pass='backbone')

        logits = helpers.extract_features_simp(train_dataloader_nosuf, model, args, forward_pass='default')
        dataset.feat_mode = False
        # dataset.one_iter_true(feats, k=int(2**(epoch/10)), max_iter=30, l2=True, index_type="ip")
        torch.cuda.synchronize()
        start = time.time()

        D, I = get_knn(args.topk_lp, feats)  # 改变近邻，改变近邻图来标签传播

        dataset.one_iter_true(feats,
                              logits,
                              D, I,
                              max_iter=30,
                              epoch=epoch,
                              num=args.num,
                              R1=args.R1,
                              R2=args.R2)

        torch.cuda.synchronize()
        end = time.time()
        t = end - start
        print(' one_iter_true time: {:.0f}min {:.0f}sec'.format(t // 60, t % 60))


        global_step, Loss = helpers.train_semi(p,  train_loader_l, train_loader_u, model, optimizer, epoch, global_step, args)


        #Evaluate
        print('Evaluate ...')
        predictions = get_predictions(p, val_dataloader, model)
        clustering_stats = hungarian_evaluate(0, predictions,
                                              class_names=val_dataset.classes,
                                              compute_confusion_matrix=True,
                                              confusion_matrix_file=os.path.join(p['selflabel_dir'],
                                                                                 'confusion_matrix.png'))
        print(clustering_stats)

        writer.add_scalar('Loss', Loss, epoch)
        writer.add_scalar('ACC', clustering_stats['ACC'], epoch)


        # 防止中途断开及时记录信息
        res = {'index': epoch + 1,
               'Loss': Loss,
               "ACC": clustering_stats['ACC'],
               "ARI": clustering_stats['ARI'],
               "NMI": clustering_stats['NMI']
               }

        from pandas.core.frame import DataFrame
        res = DataFrame(res, index=[0])
        file_name = os.path.join('output/baseline/selflabel', p['train_db_name'], args.version, path_suffix)
        file_name += '/' + str(args.P) + 'ACC1.csv'
        # print(file_name)
        res.to_csv(file_name, index=False, mode='a+', encoding='utf-8')

        results[0].append(round(Loss,3))
        results[1].append(round(clustering_stats['ACC'], 3))
        results[2].append(round(clustering_stats['ARI'], 3))
        results[3].append(round(clustering_stats['NMI'], 3))

        if best_acc < clustering_stats['ACC']:
            best_acc = clustering_stats['ACC']
            best_clustering_stats = clustering_stats
            best_model = copy.deepcopy(model)

        # Checkpoint
        # print('Checkpoint ...')
        # torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(),
        #             'epoch': epoch + 1, 'best_acc': best_acc, 'dataset': dataset},
        #              p['selflabel_checkpoint'])


    print('best_clustering:', best_clustering_stats)
    # 未保存head
    if best_model is not None:
        torch.save(best_model.module.state_dict(),
               os.path.join(p['selflabel_model'], 'model'+str(round(best_clustering_stats['ACC']*100, 2))+'.pth.tar'))

    para = p['setup'] + p['train_db_name'] + ' ct2=' + str(args.ct2) + 'num=' + str(args.num)   # the parameter name and value
    print(para)
    # print(args.wo1, args.wo2, args.wo3)
    index = np.arange(1, epoch_num + 1)
    res = {para: index,
           'Loss': results[0],
           "ACC": results[1],
           "ARI": results[2],
           "NMI": results[3]
           }
    from pandas.core.frame import DataFrame
    res = DataFrame(res)

    # file_name = 'results/ciafr10.csv'
    file_name = os.path.join('output/baseline/selflabel', p['train_db_name'], args.version, path_suffix)
    file_name += '/' + str(args.P) + 'ACC2.csv'
    res.to_csv(file_name, index=False, mode='a+', encoding='utf-8')


if __name__ == "__main__":
    main()
