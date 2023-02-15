from __future__ import division
from __future__ import print_function
import random
import argparse
import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from utils import *
from tqdm import trange
import torch.nn as nn
import torch
import math
from torch.nn.parameter import Parameter


import cvae_pretrain
from loguru import logger
import os

# Training settings
exc_path = sys.path[0]

class Args():
    def __init__(self, samples=4, concat=4, runs=100, consistency=False, dataset='cora', seed=42, no_cuda=False, epochs=10000, lr=0.005, wd1=0.01, wd2=5e-4, layer=64, weight_decay=5e-4, hidden=8, nb_heads=8, nb_heads_2=1, dropout=0.6, patience=100, dev=0, alpha=0.2, lamda=0.5, variant=False, test=False, tem=0.5, lam=1.):
        self.samples = samples
        self.concat = concat
        self.runs = runs
        self.consistency = consistency
        self.dataset= dataset
        self.seed = seed
        self.no_cuda = no_cuda
        self.epochs = epochs
        self.lr = lr
        self.wd1 = wd1
        self.wd2 = wd2
        self.layer = layer
        self.weight_decay = weight_decay
        self.hidden = hidden 
        self.nb_heads = nb_heads
        self.nb_heads_2 = nb_heads_2
        self.dropout = dropout
        self.patience = patience
        self.dev = dev
        self.alpha = alpha 
        self.lamda = lamda
        self.variant = variant
        self.test = test
        self.tem = tem
        self.lam = lam

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

class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, residual=False, variant=False):
        super(GraphConvolution, self).__init__() 
        self.variant = variant
        if self.variant:
            self.in_features = 2*in_features 
        else:
            self.in_features = in_features

        self.out_features = out_features
        self.residual = residual
        self.weight = Parameter(torch.FloatTensor(self.in_features,self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.out_features)
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj , h0 , lamda, alpha, l):
        theta = math.log(lamda/l+1)
        hi = torch.spmm(adj, input)
        if self.variant:
            support = torch.cat([hi,h0],1)
            r = (1-alpha)*hi+alpha*h0
        else:
            support = (1-alpha)*hi+alpha*h0
            r = support
        output = theta*torch.mm(support, self.weight)+(1-theta)*r
        if self.residual:
            output = output+input
        return output

class GCNII(nn.Module):
    def __init__(self, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(GCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return F.log_softmax(layer_inner, dim=1)

class GCNIIppi(nn.Module):
    def __init__(self, nfeat, nlayers,nhidden, nclass, dropout, lamda, alpha,variant):
        super(GCNIIppi, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden, nhidden,variant=variant,residual=True))
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, x, adj):
        _layers = []
        x = F.dropout(x, self.dropout, training=self.training)
        layer_inner = self.act_fn(self.fcs[0](x))
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.sig(self.fcs[-1](layer_inner))
        return layer_inner

class LAGCNII(nn.Module):
    def __init__(self, concat, nfeat, nlayers, nhidden, nclass, dropout, lamda, alpha, variant):
        super(LAGCNII, self).__init__()
        self.convs = nn.ModuleList()
        for _ in range(nlayers):
            self.convs.append(GraphConvolution(nhidden * concat, nhidden * concat, variant=variant))
        self.fcs = nn.ModuleList()
        for _ in range(concat):
            self.fcs.append(nn.Linear(nfeat, nhidden))
        self.fcs.append(nn.Linear(nhidden * concat, nclass))
        self.params1 = list(self.convs.parameters())
        self.params2 = list(self.fcs.parameters())
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        self.alpha = alpha
        self.lamda = lamda

    def forward(self, X_list, adj):
        _layers = []
        hidden_list = []
        for k, x in enumerate(X_list):
            x = F.dropout(x, self.dropout, training=self.training)
            hidden_list.append(self.act_fn(self.fcs[k](x)))
        layer_inner = torch.cat((hidden_list), dim=-1)
        _layers.append(layer_inner)
        for i,con in enumerate(self.convs):
            layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
            layer_inner = self.act_fn(con(layer_inner,adj,_layers[0],self.lamda,self.alpha,i+1))
        layer_inner = F.dropout(layer_inner, self.dropout, training=self.training)
        layer_inner = self.fcs[-1](layer_inner)
        return layer_inner
    

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

cvae_model = torch.load("{}/cave_pretrain_model/{}.pkl".format(exc_path, args.dataset))

def get_augmented_features(concat):
    X_list = []
    cvae_features = torch.tensor(features, dtype=torch.float32).to(device)
    for _ in range(concat):
        z = torch.randn([cvae_features.size(0), cvae_model.latent_size]).to(device)
        augmented_features = cvae_model.inference(z, cvae_features)
        augmented_features = cvae_pretrain.feature_tensor_normalize(augmented_features).detach()
        if args.cuda:
            X_list.append(augmented_features.to(device))
        else:
            X_list.append(augmented_features)
    return X_list


if args.cuda:
    adj_normalized = adj_normalized.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features_normalized = features_normalized.to(device)

all_val_acc = []
all_test_acc = []
for i in trange(args.runs, desc='Run Experiments'):
    model = LAGCNII(concat=args.concat+1,
                    nfeat=features.shape[1], 
                    nlayers=args.layer, 
                    nhidden=args.hidden, 
                    nclass=int(labels.max()) + 1, 
                    dropout=args.dropout, 
                    lamda=args.lamda, 
                    alpha=args.alpha, 
                    variant=args.variant).to(device)


    optimizer = optim.Adam([
                            {'params':model.params1,'weight_decay':args.wd1},
                            {'params':model.params2,'weight_decay':args.wd2},
                            ],lr=args.lr)
    if args.cuda:
        model.to(device)

    best = 999999999
    bad_counter = 0
    best_model = None
    best_X_list = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        output_list = []
        for k in range(args.samples):
            X_list = get_augmented_features(args.concat)
            output_list.append(torch.log_softmax(model(X_list+[features_normalized], adj_normalized), dim=-1))

        loss_train = 0.
        for k in range(len(output_list)):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
        
        loss_train = loss_train/len(output_list)
        if args.consistency:
            loss_consis = consis_loss(output_list)
            loss_train = loss_train + loss_consis

        loss_train.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_X_list = get_augmented_features(args.concat)
            output = model(get_augmented_features(args.concat)+[features_normalized], adj_normalized)
            output = torch.log_softmax(output, dim=1)
            loss_val = F.nll_loss(output[idx_val], labels[idx_val])
            
        logger.info(f'Epoch: {epoch+1},loss_train: {loss_train.item()}, loss_val:{loss_val.item()}, acc_val:{accuracy(output[idx_val], labels[idx_val])}')
         
        if loss_val < best:
            best = loss_val
            best_model = copy.deepcopy(model)
            best_X_list = copy.deepcopy(val_X_list)
            bad_counter = 0
        else:
            bad_counter += 1
        
        if bad_counter == args.patience:
            break
        
    #Validate and Test
    best_model.eval()
    output = best_model(best_X_list+[features_normalized], adj_normalized)
    output = torch.log_softmax(output, dim=1)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    logger.info(f'Run[{i+1}/{args.runs}] acc_val: {acc_val.item()},acc_test: {acc_test.item()}')
    all_val_acc.append(acc_val.item())
    all_test_acc.append(acc_test.item())


logger.info(f'Total: val_acc_mean:{np.mean(all_val_acc)} val_acc_std:{np.std(all_val_acc)} test_acc_mean:{np.mean(all_val_acc)} test_acc_std:{np.std(all_test_acc)}')
