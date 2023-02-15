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
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import SAGEConv
from tqdm.autonotebook import tqdm
from loguru import logger


# Training settings
exc_path = sys.path[0]

class Args():
    def __init__(self, runs=100, dataset='cora', seed=42, epochs=200, lr=0.01, weight_decay=5e-4, hidden=8, dropout=0.5):
        self.runs = runs
        self.dataset= dataset
        self.seed = seed
        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.hidden = hidden 
        self.dropout = dropout

args = Args()

#logging settings
logger.remove(handler_id=None)  
workdir = f'logs/{args.dataset}/{os.path.basename(__file__)}'
log_name = workdir + '/%s-{time}.log'
logger.add(log_name, encoding='utf-8', format="{time:YYYY-MM-DD HH:mm:ss} {level} : {message}")

#seed settings
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
args.cuda = torch.cuda.is_available()


class SAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv1(x, edge_index).relu()
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    
#caculate acccuracy
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'data', args.dataset)
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0]

if args.cuda:
    data = data.to(device)
    

all_val_acc = []
all_test_acc = []

for i in range(args.runs):
    # Model and optimizer
    model = SAGE(in_channels=dataset.num_features, hidden_channels=args.hidden, out_channels=dataset.num_classes)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.cuda:
        model.to(device)

    # Train model
    best = 999999999
    best_model = None
    
    #set bar
    pbar = tqdm(total=args.epochs)
    
    for epoch in range(args.epochs):
       
        pbar.set_description(f'Run Experiments[{i+1}/{args.runs}]')
        
        model.train()
        optimizer.zero_grad()

        output = torch.log_softmax(model(data.x, data.edge_index), dim=-1)
        loss_train = F.nll_loss(output[data.train_mask], data.y[data.train_mask])

        loss_train.backward()
        optimizer.step()

        model.eval()
        
        output = model(data.x, data.edge_index)
        output = torch.log_softmax(output, dim=1)
        loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
        
        logger.info(f'Epoch: {epoch+1},loss_train: {loss_train.item()}, loss_val:{loss_val.item()}, acc_val:{accuracy(output[data.train_mask], data.y[data.train_mask])}')

        if loss_val < best:
            best = loss_val
            best_model = copy.deepcopy(model)
        
       
        pbar.update()
    pbar.close()
        
    #Validate and Test
    best_model.eval()
    output = best_model(data.x, data.edge_index)
    output = torch.log_softmax(output, dim=1)
    acc_val = accuracy(output[data.val_mask], data.y[data.val_mask])
    acc_test = accuracy(output[data.test_mask], data.y[data.test_mask])
    logger.info(f'Run[{i+1}/{args.runs}] acc_val: {acc_val.item()},acc_test: {acc_test.item()}')
    all_val_acc.append(acc_val.item())
    all_test_acc.append(acc_test.item())
    
logger.info(f'Total: val_acc_mean:{np.mean(all_val_acc)} val_acc_std:{np.std(all_val_acc)} test_acc_mean:{np.mean(all_val_acc)} test_acc_std:{np.std(all_test_acc)}')

