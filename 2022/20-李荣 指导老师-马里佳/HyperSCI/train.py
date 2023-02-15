import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch_geometric
from sklearn.linear_model import LinearRegression

from model import HyperSCI, LossFunc, cfrnet
from util import load_data

''' Define Hyperparameter '''
parser = argparse.ArgumentParser()
parser.add_argument('--seeds', type=int, default=42, help='Random seed.')
parser.add_argument('--path', type=str, default= 'data/simulation/contact/contact_linear_alpha1_beta1_sigmoid.mat')

parser.add_argument('--exp_num', type=int, default=10, help='times of repeated executions')
parser.add_argument('--weight_decay', type=float, default=1e-2, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=1e-5, help='Initial learning rate.')
parser.add_argument('--h_dim', type=int, default=64, help='dim of hidden units.')
parser.add_argument('--g_dim', type=int, default=64, help='dim of interference representation.')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--wass', type=float, default=1e-3, help='Imbalance regularization param.')

parser.add_argument('--max_hyperedge_size', type=int, default=50, help='only keep hyperedges with size no more than this value (only valid in hypersize experiment)')
parser.add_argument('--clip', type=float, default=1., help='gradient clipping')
parser.add_argument('--exp_name', type=str, default='cfrnet', choices=['ITE', 'LR', 'cfrnet'])
parser.add_argument('--graph_model', type=str, default='hypergraph', choices=['hypergraph', 'graph'])


args = parser.parse_args()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device:', device)

result_path = 'data/results/' + args.path.split('/')[-1][:-4]
if not os.path.exists(result_path):
    os.makedirs(result_path)
result_path = result_path + '/'+ args.exp_name + '_' + args.graph_model + '/'
if not os.path.exists(result_path):
    os.makedirs(result_path)

''' Set random seeds '''
torch.manual_seed(args.seeds)
torch_geometric.seed_everything(args.seeds)


def baseline_LR(features, treatment, outcome, Y_true, idx_trn, idx_tst):
    model_1 = LinearRegression()
    model_0 = LinearRegression()
    idx_treated_trn = np.where(treatment[idx_trn] == 1)
    idx_control_trn = np.where(treatment[idx_trn] == 0)

    model_1.fit(features[idx_trn[idx_treated_trn]], outcome[idx_trn[idx_treated_trn]])
    model_0.fit(features[idx_trn[idx_control_trn]], outcome[idx_trn[idx_control_trn]])

    y_pred1_tst = model_1.predict(features[idx_tst])
    y_pred0_tst = model_0.predict(features[idx_tst])

    ITE_pred_tst = y_pred1_tst - y_pred0_tst
    ITE_true_tst = Y_true[1][idx_tst] - Y_true[0][idx_tst]

    n_select = len(idx_tst)
    ate = np.abs((ITE_pred_tst - ITE_true_tst).mean())
    pehe = math.sqrt(((ITE_pred_tst - ITE_true_tst) * (
                ITE_pred_tst - ITE_true_tst)).sum() / n_select)

    eval_results = {'pehe': pehe, 'ate': ate}

    return eval_results

def experiment_LR(features, treatment, outcome, Y_true, idx_trn_list, idx_val_list, idx_tst_list):
    t_begin = time.time()
    results_all = {'pehe': [], 'ate': []}

    for i_exp in range(0, args.exp_num):  # runs of experiments
        print("============== Experiment ", str(i_exp), " =========================")
        idx_trn = idx_trn_list[i_exp]
        idx_val = idx_val_list[i_exp]
        idx_tst = idx_tst_list[i_exp]

        eval_results_tst = baseline_LR(features.numpy(), treatment.numpy(), outcome.numpy(), Y_true.numpy(), idx_trn.numpy(), idx_tst.numpy())

        results_all['pehe'].append(float(eval_results_tst['pehe']))
        results_all['ate'].append(float(eval_results_tst['ate']))

    results_all['average_pehe'] = np.mean(np.array(results_all['pehe'], dtype=float))
    results_all['std_pehe'] = np.std(np.array(results_all['pehe'], dtype=float))
    results_all['average_ate'] = np.mean(np.array(results_all['ate'], dtype=float))
    results_all['std_ate'] = np.std(np.array(results_all['ate'], dtype=float))

    with open(result_path+'results.json', 'w', encoding='utf-8') as tf:
        json.dump(results_all, tf)

    print("============== Overall experiment results =========================")
    for k in results_all:
        if isinstance(results_all[k], list):
            print(k, ": ", results_all[k])
        else:
            print(k, f": {results_all[k]:.4f}")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

    return

def experiment_ite(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list):
    t_begin = time.time()

    results_all = {'pehe': [], 'ate': []}

    for i_exp in range(0, args.exp_num):  # 10 runs of experiments
        print("============== Experiment ", str(i_exp), " =========================")
        idx_trn = idx_trn_list[i_exp]
        idx_val = idx_val_list[i_exp]
        idx_tst = idx_tst_list[i_exp]

        # set model
        model = HyperSCI(features.shape[1], args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lossFunc = LossFunc(args.wass)

        # cuda
        model = model.to(device)
        features = features.to(device)
        treatments = treatments.to(device)
        outcomes = outcomes.to(device)
        Y_true = Y_true.to(device)
        hyperedge_index = hyperedge_index.to(device)
        idx_trn_list = [id.to(device) for id in idx_trn_list]
        idx_val_list = [id.to(device) for id in idx_val_list]
        idx_tst_list = [id.to(device) for id in idx_tst_list]

        # training
        train(args.epochs, model, optimizer, lossFunc, features, treatments, hyperedge_index, Y_true, idx_trn, idx_val, idx_tst)
        eval_results_tst = test(model, features, treatments, hyperedge_index, Y_true, idx_trn, idx_tst)

        results_all['pehe'].append(float(eval_results_tst['pehe']))
        results_all['ate'].append(float(eval_results_tst['ate']))

    results_all['average_pehe'] = np.mean(np.array(results_all['pehe'], dtype=float))
    results_all['std_pehe'] = np.std(np.array(results_all['pehe'], dtype=float))
    results_all['average_ate'] = np.mean(np.array(results_all['ate'], dtype=float))
    results_all['std_ate'] = np.std(np.array(results_all['ate'], dtype=float))

    with open(result_path+'results.json', 'w', encoding='utf-8') as tf:
        json.dump(results_all, tf)

    print("============== Overall experiment results =========================")
    for k in results_all:
        if isinstance(results_all[k], list):
            print(k, ": ", results_all[k])
        else:
            print(k, f": {results_all[k]:.4f}")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

    return

def experiment_cfrnet(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list):
    t_begin = time.time()

    results_all = {'pehe': [], 'ate': []}

    for i_exp in range(0, args.exp_num):  # 10 runs of experiments
        print("============== Experiment ", str(i_exp), " =========================")
        idx_trn = idx_trn_list[i_exp]
        idx_val = idx_val_list[i_exp]
        idx_tst = idx_tst_list[i_exp]

        # set model
        model = cfrnet(features.shape[1], args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        lossFunc = LossFunc(args.wass)

        # cuda
        model = model.to(device)
        features = features.to(device)
        treatments = treatments.to(device)
        outcomes = outcomes.to(device)
        Y_true = Y_true.to(device)
        idx_trn_list = [id.to(device) for id in idx_trn_list]
        idx_val_list = [id.to(device) for id in idx_val_list]
        idx_tst_list = [id.to(device) for id in idx_tst_list]

        # training
        train(args.epochs, model, optimizer, lossFunc, features, treatments, hyperedge_index, Y_true, idx_trn, idx_val, idx_tst)
        eval_results_tst = test(model, features, treatments, hyperedge_index, Y_true, idx_trn, idx_tst)

        results_all['pehe'].append(float(eval_results_tst['pehe']))
        results_all['ate'].append(float(eval_results_tst['ate']))

    results_all['average_pehe'] = np.mean(np.array(results_all['pehe'], dtype=float))
    results_all['std_pehe'] = np.std(np.array(results_all['pehe'], dtype=float))
    results_all['average_ate'] = np.mean(np.array(results_all['ate'], dtype=float))
    results_all['std_ate'] = np.std(np.array(results_all['ate'], dtype=float))

    with open(result_path+'results.json', 'w', encoding='utf-8') as tf:
        json.dump(results_all, tf)

    print("============== Overall experiment results =========================")
    for k in results_all:
        if isinstance(results_all[k], list):
            print(k, ": ", results_all[k])
        else:
            print(k, f": {results_all[k]:.4f}")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_begin))

    return

def report_info(epoch, time_begin, loss_results_train, eval_results_val, eval_results_tst):
    loss_train = loss_results_train['loss']
    loss_y = loss_results_train['loss_y']
    loss_b = loss_results_train['loss_b']
    pehe_val, ate_val = eval_results_val['pehe'], eval_results_val['ate']
    pehe_tst, ate_tst = eval_results_tst['pehe'], eval_results_tst['ate']

    print('Epoch: {:04d}'.format(epoch + 1),
            'loss_train: {:.4f}'.format(loss_train.item()),
            'loss_y: {:.4f}'.format(loss_y.item()),
            'loss_b: {:.4f}'.format(loss_b),
            'pehe_val: {:.4f}'.format(pehe_val),
            'ate_val: {:.4f} '.format(ate_val),
            'pehe_tst: {:.4f}'.format(pehe_tst),
            'ate_tst: {:.4f} '.format(ate_tst),
            'time: {:.4f}s'.format(time.time() - time_begin)
          )

def evaluate(Y_true, treatments, results, idx_trn, idx_select):
    y1_true = Y_true[1]
    y0_true = Y_true[0]
    y1_pred = results['y1_pred'].squeeze()
    y0_pred = results['y0_pred'].squeeze()

    # potential outcome prediction
    YF = torch.where(treatments > 0, y1_true, y0_true)

    # norm y
    ym, ys = torch.mean(YF[idx_trn]), torch.std(YF[idx_trn])
    y1_pred, y0_pred = y1_pred * ys + ym, y0_pred * ys + ym

    ITE_pred = y1_pred - y0_pred
    ITE_true = y1_true - y0_true

    # metrics
    n_select = len(idx_select)
    ate = (torch.abs((ITE_pred[idx_select] - ITE_true[idx_select]).mean())).item()
    pehe = math.sqrt(((ITE_pred[idx_select] - ITE_true[idx_select]) * (ITE_pred[idx_select] - ITE_true[idx_select])).sum().data / n_select)

    eval_results = {'pehe': pehe, 'ate': ate}

    return eval_results

def train(epochs, model, optimizer, lossFunc, features, treatments, hyperedge_index, Y_true, idx_trn, idx_val, idx_tst):
    time_begin = time.time()
    print("start training!")

    for k in range(epochs):  # epoch
        model.train()
        optimizer.zero_grad()

        # forward
        results = model(features, treatments, hyperedge_index)

        # loss
        loss_results_train = lossFunc(Y_true, treatments, results, idx_trn, idx_trn)
        loss_train = loss_results_train['loss']
        loss_train.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip) # 梯度裁减，解决梯度爆炸的问题

        if k % 50 == 0:
            # evaluate
            model.eval()
            eval_results_val = evaluate(Y_true, treatments, results, idx_trn, idx_val)
            eval_results_tst = evaluate(Y_true, treatments, results, idx_trn, idx_tst)

            report_info(k, time_begin, loss_results_train, eval_results_val, eval_results_tst)
    return

def test(model, features, treatments, hyperedge_index, Y_true, idx_trn, idx_select):
    model.eval()

    results = model(features, treatments, hyperedge_index)
    eval_results = evaluate(Y_true, treatments, results, idx_trn, idx_select)

    pehe = eval_results['pehe']
    ate = eval_results['ate']

    print('test results: ',
        'pehe_tst: {:.4f}'.format(pehe),
        'ate_tst: {:.4f} '.format(ate))

    return eval_results

if __name__ == '__main__':
    with open(result_path+'config.json', 'w', encoding='utf-8') as fp:
        json.dump(args.__dict__, fp, ensure_ascii=False)

    print('exp_name: ', args.exp_name, ' graph_model: ', args.graph_model)
    features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list = load_data(result_path, args.path, args.exp_num)  # return tensors

    # =========  Experiment 1: compare with baselines ============
    if args.exp_name == 'LR':
        experiment_LR(features, treatments, outcomes, Y_true, idx_trn_list, idx_val_list, idx_tst_list)
    elif args.exp_name == 'ITE':
        experiment_ite(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list)
    elif args.exp_name == 'cfrnet':
        experiment_cfrnet(args, features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list)
