import argparse
import csv
import datetime
import os
import shutil

import numpy as np
import scipy
import networkx as nx
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.neighbors import kneighbors_graph
from torch_geometric.data import Data, DataLoader
from Net import *
from smiles2vector import *
from utils import *

# 数据文件
# frequency_metric_file里面存放了750+9个药物和副作用的频率矩阵，矩阵具体就是下面的raw(行是药物，列是副作用)
frequency_metric_file = 'data_750+9/750+9_frequency.mat'
# 读取原始药物-副作用频率矩阵
frequency_metric = scipy.io.loadmat(frequency_metric_file)
f_metric = frequency_metric['R']
# 759种药物的分子结构
SMILES_file = 'data_750+9/drug_SMILES_759.csv'
#  drug_dict是一个字典，存放了药物的名称和对应的序号（序号应该是用来访问drug_smile的），drug_smile存放了药物的分子结构（是一个列表，每个元素也是列表）
drug_dict, drug_smile = load_drug_smile(SMILES_file)

# 筛选矩阵
blind_mask_mat_file = './data_750+9/blind_mask_mat.mat'

# 副作用特征矩阵（994*243）
side_effect_file = 'data_750+9/side_effect_label_750.mat'

dataset = 'drug_sideeffect'
# 药物特征输入维度（109维）
drug_input_dim = 109

def generateMat():
    filenames = os.listdir('data_750+9/processed')
    # 把之前生成的数据删除
    for s in filenames:
        os.remove('data_750+9/processed/' + s)

    drug_num = len(f_metric)
    # 获取筛选矩阵
    index = np.arange(0, drug_num, 1)
    np.random.shuffle(index)
    x = []
    n = int(np.ceil(drug_num / 10))
    for i in range(10):
        if i == 9:
            x.append(index.tolist())
        x.append(index[0:n].tolist())
        index = index[n:]

    # 把10份mask存储在dic中
    dic = {}
    for j in range(10):
        mask = np.ones(f_metric.shape)
        mask[x[j]] = 0
        dic['mask' + str(j)] = mask
    scipy.io.savemat(blind_mask_mat_file, dic)

def split_data():
    test_smiles = drug_smile[0:9]
    train_smiles = drug_smile[9:]
    test_frequency = f_metric[0:9]
    train_frequency = f_metric[9:]
    test_smile_graph = conver2graph(test_smiles)
    train_smile_graph = conver2graph(train_smiles)
    # 获取数据集
    train_data = myDataset(root='data_750+9', dataset=dataset + '_blind_train', drug_simles=train_smiles,
                           frequencyMat=train_frequency, simle_graph=train_smile_graph)
    test_data = myDataset(root='data_750+9', dataset=dataset + '_blind_test', drug_simles=test_smiles,
                          frequencyMat=test_frequency, simle_graph=test_smile_graph)
    return 0, train_frequency


def loss(output, frequency, lam, eps):
    x0 = torch.where(frequency == 0)
    x1 = torch.where(frequency != 0)
    loss = torch.sum((output[x1] - frequency[x1]) ** 2) \
           + lam * torch.sum((output[x0] - eps) ** 2)
    return loss


def train(device, model, train_loader, sideEffectsGraph, optimizer, DF, not_FC, lamb, eps, epoch):
    model.train()
    # 平均损失
    avg_loss = []
    sideEffectsGraph = sideEffectsGraph.to(device)
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        out, _, _ = model(data, sideEffectsGraph, DF, not_FC)
        pred = out.to(device)
        loss = loss(pred.flatten(), data.y.flatten(), lamb, eps)
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if (batch_idx + 1) % 20 == 0:
            print('{} Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(id, epoch, (batch_idx + 1) * len(data.y),
                                                                              len(train_loader.dataset), 100. * (batch_idx + 1) / len(train_loader),
                                                                              loss.item()))
    return sum(avg_loss) / len(avg_loss)


def predict(model, sideEffectsGraph, device, loder, DF, not_FC):
    model.eval()
    torch.cuda.manual_seed(42)
    total_preds = torch.Tensor()
    total_frequency = torch.Tensor()
    with torch.no_grad():
        sideEffectsGraph = sideEffectsGraph.to(device)
        for batch_idx, data in enumerate(loder):
            data = data.to(device)
            out, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            # 找出已知频率项的部分
            frequency = torch.Tensor(data.y)
            position = torch.where(frequency != 0)
            pred = out[position]
            label = frequency[position]
            total_preds = torch.cat((total_preds, pred.cpu()), 0)
            total_frequency = torch.cat((total_frequency, label.cpu()), 0)
    return total_frequency.numpy().flatten(), total_preds.numpy().flatten()

def evaluate(model, sideEffectsGraph, loder, device, DF, not_FC, result_folder):
    total_preds = torch.Tensor()
    total_frequency = torch.Tensor()
    singleDrug_auc = []
    singleDrug_aupr = []
    model.eval()
    torch.cuda.manual_seed(42)
    sideEffectsGraph = sideEffectsGraph.to(device)
    with torch.no_grad():
        for data in loder:
            frequency = data.y
            data = data.to(device)
            output, _, _ = model(data, sideEffectsGraph, DF, not_FC)
            pred = output.cpu()
            total_preds = torch.cat((total_preds, pred), 0)
            total_frequency = torch.cat((total_frequency, frequency), 0)
            pred = pred.numpy().flatten()
            frequency = (frequency.numpy().flatten() != 0).astype(int)
            singleDrug_auc.append(roc_auc_score(frequency, pred))
            singleDrug_aupr.append(average_precision_score(frequency, pred))

    pred_result = pd.DataFrame(total_preds.numpy())
    raw_result = pd.DataFrame(total_frequency.numpy())
    pred_result.to_csv(result_folder + '/blind_pred.csv', header=False, index=False)
    raw_result.to_csv(result_folder + '/blind_raw.csv', header=False, index=False)
    drugAUC = sum(singleDrug_auc) / len(singleDrug_auc)
    drugAUPR = sum(singleDrug_aupr) / len(singleDrug_aupr)
    total_preds = total_preds.numpy()
    total_label = total_frequency.numpy()
    # 是一个列表（相当于从原来的频率矩阵中挑出不为0的）
    pos = total_preds[np.where(total_label)]
    pos_label = np.ones(len(pos))
    neg = total_preds[np.where(total_label == 0)]  # 是一个列表
    neg_label = np.zeros(len(neg))
    y = np.hstack((pos, neg))  # 拼接
    y_true = np.hstack((pos_label, neg_label))
    auc_all = roc_auc_score(y_true, y)
    aupr_all = average_precision_score(y_true, y)
    Te = {}
    Te_all = {}
    Te_pairs = np.where(total_label)
    Te_pairs = np.array(Te_pairs).transpose()
    for pair in Te_pairs:
        drug_id = pair[0]
        SE_id = pair[1]
        if drug_id not in Te:
            Te[drug_id] = [SE_id]
        else:
            Te[drug_id].append(SE_id)
    shape = total_label.shape
    for i in range(shape[0]):
        Te_all[i] = [i for i in range(shape[1])]
    positions = [1, 5, 10, 15]
    map_value, auc_value, ndcg, prec, rec = evaluate_others(total_preds, Te_all, Te, positions)
    p1, p5, p10, p15 = prec[0], prec[1], prec[2], prec[3]
    r1, r5, r10, r15 = rec[0], rec[1], rec[2], rec[3]
    return auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15

def main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name, frequencyMat,
         id, result_folder, save_model, DF, not_FC, output_dim, eps, pca):
    print('\n=======================================================================================')
    print('\n第 {} 次训练：\n'.format(id))
    print('model: ', modeling.__name__)
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    print('Batch size: ', train_batch)
    print('Lambda: ', lamb)
    print('weight_decay: ', weight_decay)
    print('KNN: ', knn)
    print('metric: ', metric)
    print('tenfold: ', tenfold)
    print('DF: ', DF)
    print('not_FC: ', not_FC)
    print('output_dim: ', output_dim)
    print('Eps: ', eps)
    print('PCA: ', pca)

    model_st = modeling.__name__
    train_losses = []

    print('\nrunning on ', model_st + '_' + dataset)
    # frequency_metric_file里面存放了750+9个药物和副作用的频率矩阵，矩阵具体就是下面的raw(行是药物，列是副作用)
    processed_raw = frequency_metric_file

    if not os.path.isfile(processed_raw):
        print('Missing raw FrequencyMat, exit!!!')
        exit(1)

    # 生成副作用的graph信息
    frequencyMat = frequencyMat.T
    if pca:
        pca_ = PCA(n_components=256)
        similarity_pca = pca_.fit_transform(frequencyMat)
        print('PCA 信息保留比例： ')
        print(sum(pca_.explained_variance_ratio_))
        # A是一个矩阵，用0和1表示副作用之间是否连通（也就是副作用关联图里面的邻接矩阵）
        #  参数metric决定用什么来计算距离，mode决定了输出是0，1矩阵
        #  kneighbors_graph就是使用KNN来计算连通性
        A = kneighbors_graph(similarity_pca, knn, mode='connectivity', metric=metric, include_self=False)
    else:
        A = kneighbors_graph(frequencyMat, knn, mode='connectivity', metric=metric, include_self=False)
    G = nx.from_numpy_matrix(A.todense())  # 用邻接矩阵构造了一个关联图
    edges = []  # 用来存放副作用关联图的边信息（起点，终点）
    for (u, v) in G.edges():
        edges.append([u, v])
        edges.append([v, u])  # 无向图

    edges = np.array(edges).T  # 转置，起点一个列表，终点一个列表
    edges = torch.tensor(edges, dtype=torch.long)

    node_label = scipy.io.loadmat(side_effect_file)['node_label']  # 把994种副作用的特征（243维）拿出来，是一个994*243维的矩阵
    feat = torch.tensor(node_label, dtype=torch.float)  # 把特征矩阵转换成tensor
    sideEffectsGraph = Data(x=feat, edge_index=edges)  # 自建副作用关联图数据（特征，邻接关系）

    raw_frequency = scipy.io.loadmat(frequency_metric_file)
    raw = raw_frequency['R']  #药物-副作用频率矩阵

    # make data_WS Pytorch mini-batch processing ready
    train_data = myDataset(root='data_750+9', dataset=dataset + '_blind_train' + str(id - 1))
    train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=True)
    test_data = myDataset(root='data_750+9', dataset=dataset + '_blind_test' + str(id - 1))
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    print('CPU/GPU: ', torch.cuda.is_available())

    # 训练模型
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    model = modeling(input_dim=109, output_dim=output_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model_file_name = str(id) + 'Blind_MF_' + model_st + '_epoch=' + str(num_epoch) + '.model'
    result_log = result_folder + '/' + model_st + '_result.csv'
    loss_fig_name = str(id) + model_st + '_loss'


    for epoch in range(num_epoch):
        train_loss = train(model=model, device=device, train_loader=train_loader, optimizer=optimizer, lamb=lamb,
                           epoch=epoch + 1, log_interval=log_interval, sideEffectsGraph=sideEffectsGraph, raw=raw,
                           id=id, DF=DF, not_FC=not_FC, eps=eps)
        train_losses.append(train_loss)

    test_labels, test_preds = predict(model=model, device=device, loader=test_loader,
                                      sideEffectsGraph=sideEffectsGraph, DF=DF, not_FC=not_FC)
    ret_test = [mse(test_labels, test_preds), pearson(test_labels, test_preds), rmse(test_labels, test_preds),
                spearman(test_labels, test_preds), MAE(test_labels, test_preds)]
    test_pearsons, test_rMSE, test_spearman, test_MAE = ret_test[1], ret_test[2], ret_test[3], ret_test[4]
    auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg, p1, p5, p10, p15, r1, r5, r10, r15 = evaluate(model=model,
                                                                                                         device=device,
                                                                                                         loader=test_loader,
                                                                                                         sideEffectsGraph=sideEffectsGraph,
                                                                                                         DF=DF,
                                                                                                         not_FC=not_FC,
                                                                                                         result_folder=result_folder,
                                                                                                         id=id)
    if save_model:
        checkpointsFolder = result_folder + '/checkpoints/'
        isCheckpointExist = os.path.exists(checkpointsFolder)
        if not isCheckpointExist:
            os.makedirs(checkpointsFolder)
        torch.save(model.state_dict(), checkpointsFolder + model_file_name)

    result = [test_pearsons, test_rMSE, test_spearman, test_MAE, auc_all, aupr_all, drugAUC, drugAUPR, map_value, ndcg,
              p1, p5, p10, p15, r1, r5, r10, r15]
    with open(result_log, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(result)

    print('Test:\nPearson: {:.5f}\trMSE: {:.5f}\tSpearman: {:.5f}\tMAE: {:.5f}'.format(result[0], result[1], result[2],
                                                                                       result[3]))
    print('\tall AUC: {:.5f}\tall AUPR: {:.5f}\tdrug AUC: {:.5f}\tdrug AUPR: {:.5f}'.format(result[4], result[5],
                                                                                            result[6], result[7]))
    print('\tMAP: {:.5f}\tnDCG@10: {:.5f}'.format(map_value, ndcg))
    print('\tP@1: {:.5f}\tP@5: {:.5f}\tP@10: {:.5f}\tP@15: {:.5f}'.format(p1, p5, p10, p15))
    print('\tR@1: {:.5f}\tR@5: {:.5f}\tR@10: {:.5f}\tR@15: {:.5f}'.format(r1, r5, r10, r15))
    # train loss
    my_draw_loss(train_losses, loss_fig_name, result_folder)


if __name__ == '__main__':

    total_start = datetime.datetime.now()
    # 参数定义
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--model', type=int, required=False, default=0,
                        help='0: GAT3, 1: RGCN')
    parser.add_argument('--metric', type=int, required=False, default=0, help='0: cosine, 1: jaccard, 2: euclidean')
    parser.add_argument('--train_batch', type=int, required=False, default=10, help='Batch size training set')
    parser.add_argument('--lr', type=float, required=False, default=1e-4, help='Learning rate')
    parser.add_argument('--wd', type=float, required=False, default=0.001, help='weight_decay')
    parser.add_argument('--lamb', type=float, required=False, default=0.03, help='LAMBDA')
    parser.add_argument('--epoch', type=int, required=False, default=3000, help='Number of epoch')
    parser.add_argument('--knn', type=int, required=False, default=5, help='Number of KNN')
    parser.add_argument('--log_interval', type=int, required=False, default=20, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default='cuda:0', help='Cuda')
    parser.add_argument('--dim', type=int, required=False, default=200, help='output dim, <= 109')
    parser.add_argument('--eps', type=float, required=False, default=0.5, help='regard 0 as eps when training')

    parser.add_argument('--tenfold', action='store_true', default=False, help='use 10 folds Cross-validation ')
    parser.add_argument('--save_model', action='store_true', default=False, help='save model and features')
    parser.add_argument('--DF', action='store_true', default=False, help='use DF decoder')
    parser.add_argument('--not_FC', action='store_true', default=False, help='not use Linear layers')
    parser.add_argument('--PCA', action='store_true', default=False, help='use PCA')
    args = parser.parse_args()

    modeling = [GAT3, RGCN][args.model]
    metric = ['cosine', 'jaccard', 'euclidean'][args.metric]
    train_batch = args.train_batch
    lr = args.lr
    knn = args.knn
    num_epoch = args.epoch
    weight_decay = args.wd
    lamb = args.lamb
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    tenfold = args.tenfold
    save_model = args.save_model
    DF = args.DF
    not_FC = args.not_FC
    output_dim = args.dim
    eps = args.eps
    pca = args.PCA


    result_folder = './result_750+9/'

    if tenfold:
        result_folder += '10ICS750+9_' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
            weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
            output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + '_' + str(metric)
    else:
        result_folder += '1ICS750+9_(2)' + modeling.__name__ + '_knn=' + str(knn) + '_wd=' + str(
            weight_decay) + '_epoch=' + str(num_epoch) + '_lamb=' + str(lamb) + '_lr' + str(lr) + '_dim=' + str(
            output_dim) + '_eps=' + str(eps) + '_DF=' + str(DF) + '_PCA=' + str(pca) + '_not-FC=' + str(not_FC) + '_' + str(metric)

    isExist = os.path.exists(result_folder)
    if not isExist:
        os.makedirs(result_folder)
    else:
        # 清空原文件 添加表头
        shutil.rmtree(result_folder)
        os.makedirs(result_folder)

    result_log = result_folder + '/' + modeling.__name__ + '_result.csv'
    raw_frequency = scipy.io.loadmat(frequency_metric_file)
    raw = raw_frequency['R']

    with open(result_log, 'w', newline='') as f:
        fieldnames = ['pearson', 'rMSE', 'spearman', 'MAE', 'auc_all', 'aupr_all', 'drugAUC', 'drugAUPR', 'MAP', 'nDCG',
                      'P1', 'P5', 'P10', 'P15', 'R1', 'R5', 'R10', 'R15']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    id, frequencyMat= split_data(tenfold)
    start = datetime.datetime.now()
    main(modeling, metric, train_batch, lr, num_epoch, knn, weight_decay, lamb, log_interval, cuda_name,
         frequencyMat, id + 1, result_folder, save_model, DF, not_FC, output_dim, eps, pca)
    end = datetime.datetime.now()
    print('本次运行时间：{}\t'.format(end - start))

    data = pd.read_csv(result_log)
    L = len(data.rMSE)
    avg = [sum(data.pearson) / L, sum(data.rMSE) / L, sum(data.spearman) / L, sum(data.MAE) / L, sum(data.auc_all) / L,
           sum(data.aupr_all) / L, sum(data.drugAUC) / L, sum(data.drugAUPR) / L, sum(data.MAP) / L, sum(data.nDCG) / L,
           sum(data.P1) / L, sum(data.P5) / L, sum(data.P10) / L, sum(data.P15) / L, sum(data.R1) / L, sum(data.R5) / L,
           sum(data.R10) / L, sum(data.R15) / L]
    print('\n\tavg pearson: {:.4f}\tavg rMSE: {:.4f}\tavg spearman: {:.4f}\tavg MAE: {:.4f}'.format(avg[0], avg[1],
                                                                                                    avg[2], avg[3]))
    print('\tavg all AUC: {:.4f}\tavg all AUPR: {:.4f}\tavg drug AUC: {:.4f}\tavg drug AUPR: {:.4f}'.format(avg[4],
                                                                                                            avg[5],
                                                                                                            avg[6],
                                                                                                            avg[7]))
    print('\tavg MAP: {:.4f}\tavg nDCG@10: {:.4f}'.format(avg[8], avg[9]))
    print('\tavg P@1: {:.4f}\tavg P@5: {:.4f}\tavg P@10: {:.4f}\tavg P@15: {:.4f}'.format(avg[10], avg[11], avg[12],
                                                                                          avg[13]))
    print('\tavg R@1: {:.4f}\tavg R@5: {:.4f}\tavg R@10: {:.4f}\tavg R@15: {:.4f}'.format(avg[14], avg[15], avg[16],
                                                                                          avg[17]))
    with open(result_log, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['avg'])
        writer.writerow(avg)
    total_end = datetime.datetime.now()
    print('总体运行时间：{}\t'.format(total_end - total_start))