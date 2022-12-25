import os
import sys
import random
import argparse
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import cvae_pretrain 
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from tqdm import trange
from utils import load_data, accuracy, normalize_features
from loguru import logger
import os

# Training settings
exc_path = sys.path[0]

class Args():
    def __init__(self, samples=4, concat=4, runs=100, dataset='cora', seed=42, epochs=200, lr=0.01, weight_decay=5e-4, hidden=8, dropout=0.5, tem=0.5, lam=1., alpha=0.2):
        self.samples = samples
        self.concat = concat
        self.runs = runs
        self.dataset= dataset
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden = hidden 
        self.dropout = dropout
        self.tem = tem
        self.lam = lam
        self.alpha = alpha

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


class LASAGE(torch.nn.Module):
    def __init__(self, concat, nfeat, nclass):
        super().__init__()
        
        self.gcn1_list = nn.ModuleList()
        for _ in range(concat):
            self.gcn1_list.append(SAGEConv(nfeat, args.hidden))

        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = SAGEConv(args.hidden*concat, nclass)

    def forward(self, x_list, edge_index):
        hidden_list = []
        for k, con in enumerate(self.gcn1_list):
            x = F.dropout(x_list[k], p=args.dropout, training=self.training)
            hidden_list.append(F.elu(con(x, edge_index), alpha=args.alpha))

        x = torch.cat((hidden_list), dim=-1)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x



# Load data
adj, features, idx_train, idx_val, idx_test, labels = load_data(args.dataset)

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', args.dataset)
data = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())[0]


# Normalize adj and features
features = features.toarray()
features_normalized = normalize_features(features)

# To PyTorch Tensor
labels = torch.LongTensor(labels)
labels = torch.max(labels, dim=1)[1]
features_normalized = torch.FloatTensor(features_normalized)
idx_train = torch.LongTensor(idx_train)
idx_val = torch.LongTensor(idx_val)
idx_test = torch.LongTensor(idx_test)

cvae_model = torch.load("{}/cave_pretrain_model/{}.pkl".format(exc_path, args.dataset), map_location=device)

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
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)
    features_normalized = features_normalized.to(device)
    data = data.to(device)


all_val_acc = []
all_test_acc = []
for i in trange(args.runs, desc='Run Experiments'):
    # Model and optimizer
    model = LASAGE(concat=args.concat+1,
                  nfeat=features.shape[1], 
                  nclass=int(labels.max().item()) + 1)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.to(device)

    # Train model
    best = 999999999
    best_model = None
    best_X_list = None
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        output_list = []
        for k in range(args.samples):
            X_list = get_augmented_features(args.concat)
            output_list.append(torch.log_softmax(model(X_list+[features_normalized], data.edge_index), dim=-1))

        loss_train = 0.
        for k in range(len(output_list)):
            loss_train += F.nll_loss(output_list[k][idx_train], labels[idx_train])
        
        loss_train = loss_train/len(output_list)

        loss_consis = consis_loss(output_list)
        loss_train = loss_train + loss_consis

        loss_train.backward()
        optimizer.step()

        model.eval()
        val_X_list = get_augmented_features(args.concat)
        output = model(val_X_list+[features_normalized], data.edge_index)
        output = torch.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])

        logger.info(f'Epoch: {epoch+1},loss_train: {loss_train.item()}, loss_val:{loss_val.item()}, acc_val:{accuracy(output[idx_val], labels[idx_val])}')
            
        if loss_val < best:
            best = loss_val
            best_model = copy.deepcopy(model)
            best_X_list = copy.deepcopy(val_X_list)
       
        
    #Validate and Test
    best_model.eval()
    output = best_model(best_X_list+[features_normalized], data.edge_index)
    output = torch.log_softmax(output, dim=1)
    acc_val = accuracy(output[idx_val], labels[idx_val])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    logger.info(f'Run[{i+1}/{args.runs}] acc_val: {acc_val.item()},acc_test: {acc_test.item()}')
    all_val_acc.append(acc_val.item())
    all_test_acc.append(acc_test.item())

logger.info(f'Total: val_acc_mean:{np.mean(all_val_acc)} val_acc_std:{np.std(all_val_acc)} test_acc_mean:{np.mean(all_val_acc)} test_acc_std:{np.std(all_test_acc)}')

