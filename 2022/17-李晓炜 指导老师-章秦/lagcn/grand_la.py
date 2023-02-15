from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import sys
import random
import scipy.sparse as sp
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim
import cvae_pretrain

from tqdm import trange
from utils import load_data, accuracy, normalize_adj, normalize_features, sparse_mx_to_torch_sparse_tensor
from loguru import logger
import os
import math

import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np
from torch.nn.modules.module import Module

exc_path = sys.path[0]

# Training settings

class Args():
    def __init__(self, sample=4, runs=100, dataset='cora', seed=42, epochs=5000, lr=0.01, weight_decay=5e-4, hidden=8, tem=0.5, lam=1., input_droprate=0.5, hidden_droprate=0.5, dropnode_rate=0.5, patience=100, order=5, use_bn=False):
        self.sample = sample
        self.runs = runs
        self.dataset= dataset
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden = hidden 
        self.tem = tem
        self.lam = lam
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.dropnode_rate = dropnode_rate
        self.patience = patience
        self.order = order
        self.use_bn = use_bn

args = Args()


torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

#logging settings
logger.remove(handler_id=None)  
workdir = f'logs/{args.dataset}/{os.path.basename(__file__)}'
log_name = workdir + '/%s-{time}.log'
logger.add(log_name, encoding='utf-8', format="{time:YYYY-MM-DD HH:mm:ss} {level} : {message}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()

class MLPLayer(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MLPLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.normal_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

    

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, use_bn=False):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn

    def forward(self, x, adj):

        if self.use_bn:
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        
        x = F.relu(self.gc1(x, adj))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.gc2(x, adj)

        return x

class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nclass, input_droprate, hidden_droprate, is_cuda=True, use_bn =False):
        super(MLP, self).__init__()

        self.layer1 = MLPLayer(nfeat, nhid)
        self.layer2 = MLPLayer(nhid, nclass)

        self.input_droprate = input_droprate
        self.hidden_droprate = hidden_droprate
        self.is_cuda = is_cuda
        self.bn1 = nn.BatchNorm1d(nfeat)
        self.bn2 = nn.BatchNorm1d(nhid)
        self.use_bn = use_bn
        
    def forward(self, x):
         
        if self.use_bn: 
            x = self.bn1(x)
        x = F.dropout(x, self.input_droprate, training=self.training)
        x = F.relu(self.layer1(x))
        if self.use_bn:
            x = self.bn2(x)
        x = F.dropout(x, self.hidden_droprate, training=self.training)
        x = self.layer2(x)

        return x


    

# Load data
adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)

# Normalize adj and features
features = features.toarray()
adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0])) 
features_normalized = normalize_features(features)

# To PyTorch Tensor
labels = torch.LongTensor(labels)
labels = torch.max(labels, dim=1)[1]
features_normalized = torch.FloatTensor(features_normalized)
adj_normalized = sparse_mx_to_torch_sparse_tensor(adj_normalized)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

Augmented_X = cvae_pretrain.feature_tensor_normalize(torch.load("{}/cave_pretrain_feature/{}_features.pt".format(exc_path, args.dataset)))


def propagate(feature, A, order):
    #feature = F.dropout(feature, args.dropout, training=training)
    x = feature
    y = feature
    for i in range(order):
        x = torch.spmm(A, x).detach_()
        #print(y.add_(x))
        y.add_(x)
        
    return y.div_(order+1.0).detach_()


def rand_prop(features, A, training):
    n = features.shape[0]
    drop_rate = args.dropnode_rate
    drop_rates = torch.FloatTensor(np.ones(n) * drop_rate)
    
    if training:
            
        masks = torch.bernoulli(1. - drop_rates).unsqueeze(1)
        if args.cuda:
            masks = masks.cuda()

        #features = masks.cuda() * features
        features = masks * features
            
    else:
            
        features = features * (1. - drop_rate)
    features = propagate(features, A, args.order)    
    return features


def consis_loss(logps, temp=args.tem):
    ps = [torch.exp(p) for p in logps]
    sum_p = 0.
    for p in ps:
        sum_p = sum_p + p
    avg_p = sum_p/len(ps)
    #p2 = torch.exp(logp2)
    
    sharp_p = (torch.pow(avg_p, 1./temp) / torch.sum(torch.pow(avg_p, 1./temp), dim=1, keepdim=True)).detach()
    loss = 0.
    for p in ps:
        loss += torch.mean((p-sharp_p).pow(2).sum(1))
    loss = loss/len(ps)
    return args.lam * loss


if args.cuda:
    features_normalized = features_normalized.to(device)
    adj_normalized = adj_normalized.to(device)
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

all_val_acc = []
all_test_acc = []
for i in trange(args.runs, desc='Run Experiments'):
    # Model and optimizer
    model = MLP(nfeat=features.shape[1],
                nhid=args.hidden,
                nclass=labels.max().item() + 1,
                input_droprate=args.input_droprate,
                hidden_droprate=args.hidden_droprate,
                use_bn = args.use_bn)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.cuda()


    loss_values = []
    acc_values = []
    bad_counter = 0
    loss_best = np.inf
    acc_best = 0.0

    loss_mn = np.inf
    acc_mx = 0.0

    best_epoch = 0
    best_model = None
    for epoch in range(args.epochs):
        X = features_normalized
        A = adj_normalized

        model.train()
        optimizer.zero_grad()
        X_list = []
        K = args.sample
        for k in range(K):
            X_list.append(rand_prop(X, A, training=True))
            X_list.append(rand_prop(Augmented_X, A, training=True))

        output_list = []
        for k in range(len(X_list)):
            output_list.append(torch.log_softmax(model(X_list[k]), dim=-1))

        loss_train = 0.
        for k in range(len(X_list)):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
        
        loss_train = loss_train/len(X_list)
        loss_consis = consis_loss(output_list)

        loss_train = loss_train + loss_consis
        acc_train = accuracy(output_list[0][idx_train], labels[idx_train])
        loss_train.backward()
        optimizer.step()

        model.eval()
        X = rand_prop(X, A, training=False)
        output = model(X)
        output = torch.log_softmax(output, dim=-1)

        loss_val = F.nll_loss(output[idx_val], labels[idx_val]) 
        acc_val = accuracy(output[idx_val], labels[idx_val])

        loss_values.append(loss_val.item())
        acc_values.append(acc_val.item())

        logger.info(f'Epoch: {epoch+1},loss_train: {loss_train.item()}, loss_val:{loss_val.item()}, acc_val:{accuracy(output[idx_val], labels[idx_val])}')
         
        
        if loss_values[-1] <= loss_mn or acc_values[-1] >= acc_mx:
            if loss_values[-1] <= loss_best: #and acc_values[-1] >= acc_best:
                loss_best = loss_values[-1]
                acc_best = acc_values[-1]
                best_epoch = epoch
                best_X = copy.deepcopy(X)
                best_model = copy.deepcopy(model)

            loss_mn = np.min((loss_values[-1], loss_mn))
            acc_mx = np.max((acc_values[-1], acc_mx))
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter == args.patience:
            break
        
    best_model.eval()
    X = rand_prop(features_normalized, A, training=False)
    output = best_model(X)
    output = torch.log_softmax(output, dim=-1)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    all_val_acc.append(acc_val.item())
    logger.info(f'Run[{i+1}/{args.runs}] acc_val: {acc_val.item()},acc_test: {acc_test.item()}')
    all_test_acc.append(acc_test.item())

logger.info(f'Total: val_acc_mean:{np.mean(all_val_acc)} val_acc_std:{np.std(all_val_acc)} test_acc_mean:{np.mean(all_val_acc)} test_acc_std:{np.std(all_test_acc)}')
