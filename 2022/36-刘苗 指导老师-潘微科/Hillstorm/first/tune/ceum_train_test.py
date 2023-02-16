import random
import numpy as np
import optuna
from optuna.samplers import TPESampler
from sklift.metrics import uplift_auc_score, qini_auc_score, uplift_at_k
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from utils.helper import weighted_average_uplift
import warnings
warnings.filterwarnings("ignore")

class Ceum(nn.Module):

    def __init__(self, input_dim, input_dim1, h_dim, is_self, act_type='relu', margin=0.5, margin1=0.5, alpha=0.1,
                 beta=0.1):
        super(Ceum, self).__init__()
        self.is_self = is_self
        self.s_fc1 = nn.Linear(input_dim + 1, h_dim)
        self.s_fc2 = nn.Linear(h_dim, h_dim)
        self.s_fc3 = nn.Linear(h_dim, h_dim // 2)
        self.s_fc4 = nn.Linear(h_dim // 2, h_dim // 4)

        out_dim = h_dim // 4

        if self.is_self:
            self.s_fc5 = nn.Linear(h_dim / 4, h_dim // 8)
            out_dim = h_dim // 8

        self.s_logit = nn.Linear(out_dim, 1)

        # activation function
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

        # Hyperparameter
        self.margin = margin
        self.margin1 = margin1
        self.alpha = alpha
        self.beta = beta

    def forward(self, feature_list, is_treat, feature_list1, is_treat1):
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

        is_treat1 = torch.unsqueeze(is_treat1, dim=1)
        xt1 = torch.cat((feature_list1, is_treat1), dim=1)
        s_last1 = self.act(self.s_fc4(self.act(self.s_fc3(self.act(self.s_fc2(self.act(self.s_fc1(xt1))))))))
        if self.is_self:
            s_last1 = self.act(self.s_fc5(s_last1))
        s_logit1 = self.s_logit(s_last1)
        s_prob1 = torch.sigmoid(s_logit1)
        _xt1 = torch.cat((feature_list1, (1 - is_treat1)), dim=1)
        _s_last1 = self.act(self.s_fc4(self.act(self.s_fc3(self.act(self.s_fc2(self.act(self.s_fc1(_xt1))))))))
        if self.is_self:
            _s_last = self.act(self.s_fc5(_s_last1))
        _s_logit1 = self.s_logit(_s_last1)
        _s_prob1 = torch.sigmoid(_s_logit1)
        y10 = is_treat1 * _s_prob1 + (1 - is_treat1) * s_prob1
        y11 = is_treat1 * s_prob1 + (1 - is_treat1) * _s_prob1

        return s_logit, y1 - y0, s_logit1, y11 - y10

    def calculate_loss(self, feature_list, is_treat, label_list, feature_list1, is_treat1, label_list1):
        y_true = torch.unsqueeze(label_list, dim=1)
        t_true = torch.unsqueeze(is_treat, 1)
        s_logit, uplift, s_logit1, uplift1 = self.forward(feature_list, is_treat, feature_list1, is_treat1)
        criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        l_ce = criterion(s_logit, y_true)

        y_diff = torch.logical_not(torch.logical_xor(t_true, y_true))
        y_diff = torch.where(y_diff == True, torch.tensor(1, dtype=torch.float), torch.tensor(-1, dtype=torch.float))

        l_diff = torch.mean(self.margin - y_diff * (uplift - uplift1))
        l_ate = -self.margin1 + torch.square(torch.mean(y_true * t_true) - torch.mean(y_true * (1 - t_true)) - torch.mean(uplift))
        loss = l_ce + self.alpha * l_diff + self.beta * l_ate

        return loss

class WrapperModel(nn.Module):
    def __init__(self, model):
        super(WrapperModel, self).__init__()
        self.model = model

    def forward(self, feature_list, is_treat, label_list, feature_list1, is_treat1, label_list1):
        final_output = self.model.calculate_loss(feature_list, is_treat, label_list, feature_list1, is_treat1, label_list1)
        return final_output

    def predict(self, feature_list, is_treat, zhanwei, zhanwei1):
        _, predict_out, _, _ = self.model.forward(feature_list, is_treat, zhanwei, zhanwei1)
        return predict_out

def valid(model, valid_dataloader, metric):
    model.eval()
    predictions = []
    true_labels = []
    is_treatment = []

    for step, (X, T, valid_label) in enumerate(valid_dataloader):
        model.eval()
        uplift = model.predict(X, T, X, T)
        uplift = uplift.squeeze()
        predictions.extend(uplift.detach().numpy())
        true_labels.extend(valid_label.numpy())
        is_treatment.extend(T.numpy())

    true_labels = np.array(true_labels)
    predictions = np.array(predictions)
    is_treatment = np.array(is_treatment)

    # 通过整个样本的抬升来计算前k次观测的抬升
    u_at_k = uplift_at_k(true_labels, predictions, is_treatment, strategy='overall', k=0.3)

    # 根据预测得分计算Qini曲线下的标准化面积(也称为Qini系数)
    qini_coef = qini_auc_score(true_labels, predictions, is_treatment)

    # 根据预测得分计算上升曲线下的标准化面积
    uplift_auc = uplift_auc_score(true_labels, predictions, is_treatment)

    # 加权平均提升
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

    return valid_metric, u_at_k, qini_coef, uplift_auc, wau


def collote_fn(batch_samples):
    batch_input, batch_label, batch_is_treatment,batch_input1, batch_label1, batch_is_treatment1 = [], [], [],[],[],[]
    for sample in batch_samples:
        batch_input.append(sample[:8])
        batch_is_treatment.append(sample[8])
        batch_label.append(sample[9])
        batch_input1.append(sample[10:18])
        batch_is_treatment1.append(sample[18])
        batch_label1.append(sample[19])
    input_list = torch.tensor(batch_input)
    label_list = torch.tensor(batch_label)
    is_treatment_label_list = torch.tensor(batch_is_treatment)
    input_list1 = torch.tensor(batch_input1)
    label_list1 = torch.tensor(batch_label1)
    is_treatment_label_list1 = torch.tensor(batch_is_treatment1)
    return input_list, is_treatment_label_list, label_list,input_list1, is_treatment_label_list1, label_list1

def collote_fn1(batch_samples):
    batch_input, batch_label, batch_is_treatment = [], [], []
    for sample in batch_samples:
        batch_input.append(sample[:8])
        batch_label.append(sample[9])
        batch_is_treatment.append(sample[8])
    input_list = torch.tensor(batch_input)
    label_list = torch.tensor(batch_label)
    is_treatment_label_list = torch.tensor(batch_is_treatment)

    return input_list, is_treatment_label_list, label_list


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

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def objective(trial):
    # sample a set of hyperparameters.
    rank = trial.suggest_categorical('rank', [32, 64, 128])
    lamb = trial.suggest_categorical('lambda', [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    batch_size = trial.suggest_categorical('batch_size', [256, 512, 1024, 2048])
    learning_rate = trial.suggest_categorical('learning_rate', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1])
    margin = trial.suggest_categorical('margin', [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    margin1 = trial.suggest_categorical('margin1', [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    alpha = trial.suggest_categorical('alpha', [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    beta = trial.suggest_categorical('beta', [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])

    # trail:80
    #  0.0232058245442254 and parameters: {'rank': 64, 'lambda': 1e-05, 'batch_size': 512,
    #  'learning_rate': 0.0001, 'margin': 0.8, 'margin1': 0.6, 'alpha': 0.3, 'beta': 0.3}
    # epoch:17 loss:17.62111 avg_loss:0.55066
    # u_at_k:0.06550 qini_coef:0.04707 uplift_auc:0.02321 wau:0.03849


    # rank = trial.suggest_categorical('rank', [32])
    # lamb = trial.suggest_categorical('lambda', [1e-2])
    # batch_size = trial.suggest_categorical('batch_size', [256])
    # learning_rate = trial.suggest_categorical('learning_rate', [0.005])
    # margin = trial.suggest_categorical('margin', [0.2])
    # margin1 = trial.suggest_categorical('margin1', [0.7])
    # alpha = trial.suggest_categorical('alpha', [0.7])
    # beta = trial.suggest_categorical('beta', [0.5])

    setup_seed(seed)

    contrastive_pair_set = CRITEO('../contrastive_pair_Hillstrom.npy')
    train_dataloader = DataLoader(contrastive_pair_set, batch_size=batch_size, collate_fn=collote_fn)

    test_data_set = CRITEO('../test_Hillstrom.npy')
    valid_dataloader = DataLoader(test_data_set, batch_size=2048, collate_fn=collote_fn1)

    # model
    model = Ceum(input_dim=8, input_dim1=8, h_dim=rank, is_self=False, act_type="relu", margin=margin,
                 margin1=margin1, alpha=alpha, beta=beta)

    model = WrapperModel(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=lamb)

    best_valid_metric = 0
    result_early_stop = 0
    num_epoch = 20
    for epoch in range(num_epoch):
        tr_loss = 0
        tr_steps = 0
        for step, (X, T, label, X1, T1, label1) in enumerate(train_dataloader):
            tr_steps += 1
            loss = model(X, T, label, X1, T1, label1)
            model.train()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()
        model.eval()
        valid_metric, u_at_k, qini_coef, uplift_auc, wau = valid(model, valid_dataloader, 'AUUC')
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
    n_trials = 80
    setup_seed(seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    study = optuna.create_study(sampler=TPESampler(seed=seed), direction='maximize')
    study.optimize(objective, n_trials=n_trials)
