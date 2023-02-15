import random
import logging
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_at_k

import optuna
from optuna.samplers import TPESampler

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from utils.helper import weighted_average_uplift
import warnings
warnings.filterwarnings("ignore")

class SLearner(nn.Module):
    def __init__(self, input_dim, h_dim, is_self, act_type='elu'):
        super(SLearner, self).__init__()
        self.is_self = is_self
        self.s_fc1 = nn.Linear(input_dim + 1, h_dim)
        self.s_fc2 = nn.Linear(h_dim, h_dim)
        self.s_fc3 = nn.Linear(h_dim, h_dim // 2) # ‘//’ 表示得到的得数向下取整
        self.s_fc4 = nn.Linear(h_dim // 2, h_dim // 4)# 四个全连接层
        out_dim = h_dim // 4# 输出层维度
        if self.is_self:
            self.s_fc5 = nn.Linear(h_dim / 4, h_dim // 8)  #如果得到治疗则多增加一个全连接层，重新设置输出层维度
            out_dim = h_dim // 8
        self.s_logit = nn.Linear(out_dim, 1)  
        # activation function  #激活函数
        if act_type == 'sigmoid':
            self.act = nn.Sigmoid()
        elif act_type == 'tanh':
            self.act = nn.Tanh()
        elif act_type == 'relu':
            self.act = nn.ReLU()
        elif act_type == 'elu':
            self.act = nn.ELU()
        else:
            raise RuntimeError('unknown act_type {0}'.format(act_type))
    def forward(self, feature_list, is_treat ):# 前向传播
        is_treat = torch.unsqueeze(is_treat, dim=1)
        xt = torch.cat((feature_list, is_treat), dim=1)                                                       
        s_last = self.act(self.s_fc4(self.act(self.s_fc3(self.act(self.s_fc2(self.act(self.s_fc1(xt))))))))
        if self.is_self:
            s_last = self.act(self.s_fc5(s_last))                                                         
        s_logit = self.s_logit(s_last)
        s_prob = torch.sigmoid(s_logit)
        _xt = torch.cat((feature_list, (1 - is_treat)), dim=1)
        _s_last = self.act(self.s_fc4(self.act(self.s_fc3(self.act(self.s_fc2(self.act(self.s_fc1(_xt))))))))
        if self.is_self:
            _s_last = self.act(self.s_fc5(_s_last))
        _s_logit = self.s_logit(_s_last)
        _s_prob = torch.sigmoid(_s_logit)
        y0 = is_treat * _s_prob + (1 - is_treat) * s_prob 
        y1 = is_treat * s_prob + (1 - is_treat) * _s_prob                                                             
        return s_logit, y1 - y0

    def calculate_loss(self, feature_list, is_treat, label_list):
        y_true = torch.unsqueeze(label_list, dim=1)
        s_logit, uplift  = self.forward(feature_list, is_treat )
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        loss = criterion(s_logit, y_true)
        return loss

class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def forward(self, feature_list, is_treat, label_list ):
        final_output = self.model.calculate_loss(feature_list, is_treat, label_list )
        return final_output

def valid(model, valid_dataloader, device, metric):
    model.eval()
    predictions = []
    true_labels = []
    is_treatment = []

    for step, (X, T, valid_label) in enumerate(valid_dataloader):
        model.eval()

        feature_list = X.to(device)
        is_treat = T.to(device)
        label_list = valid_label.to(device)

        _, u_tau= model.model.forward(feature_list, is_treat )
        uplift = u_tau.squeeze()

        predictions.extend(uplift.detach().cpu().numpy())
        true_labels.extend(label_list.detach().cpu().numpy())
        is_treatment.extend(is_treat.detach().cpu().numpy())

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    is_treatment = np.array(is_treatment)

    # Compute uplift at first k observations by uplift of the total sample
    u_at_k = uplift_at_k(true_labels, predictions, is_treatment, strategy='overall', k=0.3)

    # Compute normalized Area Under the Qini curve (aka Qini coefficient) from prediction scores
    qini_coef = qini_auc_score(true_labels, predictions, is_treatment)

    # Compute normalized Area Under the Uplift Curve from prediction scores
    uplift_auc = uplift_auc_score(true_labels, predictions, is_treatment)

    # Weighted average uplift
    wau = weighted_average_uplift(true_labels, predictions, is_treatment, strategy='overall')

    valid_result = [u_at_k, qini_coef, uplift_auc, wau]

    if metric == "AUUC":
        valid_metric = uplift_auc
    elif metric == "QINI":
        valid_metric = qini_coef
    elif metric == 'WAU':
        valid_metric = wau
    else:
        valid_metric = u_at_k
    # logger.info("Valid results: {}".format(valid_result))
    return valid_metric, u_at_k, qini_coef, uplift_auc, wau

def collote_fn(batch_samples):
    batch_input, batch_label, batch_is_treatment = [], [], []
    for sample in batch_samples:
        batch_input.append(sample[:12])
        batch_label.append(sample[13])
        batch_is_treatment.append(sample[12])
    input_list = torch.tensor(batch_input)
    label_list = torch.tensor(batch_label)
    is_treatment_label_list = torch.tensor(batch_is_treatment)

    return input_list, is_treatment_label_list, label_list


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

class CRITEO(Dataset):
    def __init__(self, data_file):
        self.data = self.load_data(data_file)

    def load_data(self, data_file):
        data_matrix = np.load(data_file)
        Data = np.float32(data_matrix)
        return Data

    def __len__(self):
        return np.shape(self.data)[0]

    def __getitem__(self, idx):
        return self.data[idx]

def objective(trial):
    # sample a set of hyperparameters.
    rank = trial.suggest_categorical('rank', [32, 64, 128])
    lamb = trial.suggest_categorical('lambda', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
    learning_rate = trial.suggest_categorical('learning_rate', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])

    # trial:20
    # 0.006300515215403181 and parameters: {'rank': 64, 'lambda': 0.001, 'batch_size': 2048, 'learning_rate': 0.005}.    # epoch: 4
    # epoch:2 loss:0.71101 avg_loss:0.01481
    # u_at_k:0.00288 qini_coef:0.17964 uplift_auc:0.00630 wau:0.00065
    #

    # rank = trial.suggest_categorical('rank', [128])
    # lamb = trial.suggest_categorical('lambda', [1e-4])
    # batch_size = trial.suggest_categorical('batch_size', [256])
    # learning_rate = trial.suggest_categorical('learning_rate', [ 0.001])
    setup_seed(seed)
    # parameter settings
    train_file = "../train_criteo.npy"
    valid_file = "../test_criteo.npy"
    # model
    model = SLearner(input_dim=12, h_dim=rank, is_self=False, act_type="elu")
    model = WrapperModel(model).to(device)
    # data
    train_data = CRITEO(train_file)
    valid_data = CRITEO(valid_file)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, collate_fn=collote_fn)
    valid_dataloader = DataLoader(valid_data, batch_size=2048, collate_fn=collote_fn)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=lamb)
    best_valid_metric = 0
    result_early_stop = 0
    num_epoch = 20
    for epoch in range(num_epoch):
        tr_loss = 0
        tr_steps = 0
        for step, (X, T, label) in enumerate(train_dataloader):
            tr_steps += 1
            feature_list = X.to(device)
            is_treat = T.to(device)
            label_list = label.to(device)
            loss = model(feature_list, is_treat, label_list)
            model.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        model.eval()
        valid_metric, u_at_k, qini_coef, uplift_auc, wau = valid(model, valid_dataloader,device, 'AUUC')
        print('epoch:%d loss:%.5f avg_loss:%.5f u_at_k:%.5f qini_coef:%.5f uplift_auc:%.5f wau:%.5f'
              % (epoch, tr_loss, tr_loss / tr_steps, u_at_k, qini_coef, uplift_auc, wau))
        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            result_early_stop = 0
        else:
            result_early_stop += 1
            if result_early_stop > 5:
                 break
    return best_valid_metric

if __name__ == "__main__":
    # parameter settings
    seed = 0
    n_trials = 20
    setup_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
    study.optimize(objective, n_trials=n_trials)
