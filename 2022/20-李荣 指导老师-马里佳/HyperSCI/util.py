import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_scatter import scatter_add
import scipy.io as sio
from matplotlib import rc
import matplotlib

# 使LaTex字体与常规字体大小相同
rc('mathtext', default='regular')
font_sz = 28
matplotlib.rcParams['font.family'] = 'sans-serif' # 用于显示字体的名字，sans-serif通用字体族
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'
matplotlib.rcParams.update({'font.size': font_sz})

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def draw_bar(x, y, x_label, y_label=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.bar(x, y)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

def draw_freq(data, x_label=None, bool_discrete = False, save_path=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(data, bins=50)

    plt.xlabel(x_label)
    plt.ylabel("Frequency")
    plt.show()


    # Find at most 10 ticks on the y-axis
    if not bool_discrete:
        max_xticks = 10
        xloc = plt.MaxNLocator(max_xticks)
        ax.xaxis.set_major_locator(xloc)

    if save_path != None:
        plt.savefig(save_path)


'''
info of nodes & Hyperedges
'n': number of nodes
'm': number of Hyperedges
'm>2': number of Hyperedges (has nodes number > 2) 
'average_hyperedge_size': ave_hyperedge_size
'min_hyperedge_size': min_hyperedge_size, 
'max_hyperedge_size': max_hyperedge_size,
'average_degree': ave_node_degree, 
'max_degree': max_node_degree, 
'min_degree': min_node_degree
'''
def hypergraph_stats(hyperedge_index, n):
    # hyperedge size
    unique_edge, counts_edge = np.unique(hyperedge_index[1], return_counts=True)  # edgeid, size
    ave_hyperedge_size = np.mean(counts_edge)
    max_hyperedge_size = np.max(counts_edge)
    min_hyperedge_size = np.min(counts_edge)
    m = len(unique_edge)

    sz, ct = np.unique(counts_edge, return_counts=True)  # hyperedgesize, count
    counts_edge_2 = ct[np.where(sz==2)][0]

    # node degree
    unique_node, counts_node = np.unique(hyperedge_index[0], return_counts=True)  # nodeid, degree
    ave_degree = np.mean(counts_node)
    max_degree = np.max(counts_node)
    min_degree = np.min(counts_node)
    statistics = {'n': n, 'm': m, 'm>2': m-counts_edge_2,
                  'average_hyperedge_size': ave_hyperedge_size, 'min_hyperedge_size': min_hyperedge_size, 'max_hyperedge_size': max_hyperedge_size,
                  'average_degree': ave_degree, 'max_degree': max_degree, 'min_degree': min_degree}
    return statistics

def non_linear(x, type='raw'):
    if type == 'raw':
        return x
    if type == 'sigmoid':
        func = torch.nn.Sigmoid()
    elif type == 'tanh':
        func = torch.nn.Tanh()
    elif type == 'relu':
        func = torch.nn.ReLU()
    elif type == 'leaky_relu':
        func = torch.nn.LeakyReLU()

    return func(x)

def simulate_outcome_linear(features, hyperedge_index, treatments, alpha=1.0, beta=1.0, nonlinear_type='sigmoid'):
    features.to(device)
    hyperedge_index = torch.tensor(hyperedge_index).type(torch.long)
    treatments = torch.tensor(treatments)

    n = features.shape[0]
    dim_size = features.shape[1]
    y_C = 10
    c_ft = 5
    w_noise = 1.0

    w0 = torch.randn(dim_size).to(device)
    f_y00 = torch.matmul(features, w0.reshape(-1)).to(device)

    w1 = torch.randn(dim_size).to(device)
    ites = torch.matmul(features, w1).reshape(-1) + c_ft
    ites.to(device)
    f_t = ites * treatments

    f_t_he_ave = scatter_add(f_t[hyperedge_index[0]], hyperedge_index[1], dim=0).to(device)
    Ne = scatter_add(torch.ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0).to(device)
    f_t_he_ave = f_t_he_ave/Ne
    f_t_he_ave = non_linear(f_t_he_ave, nonlinear_type)
    f_s = scatter_add(f_t_he_ave[hyperedge_index[1]], hyperedge_index[0], dim=0).to(device)
    Ee = scatter_add(torch.ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0).to(device)
    f_s = f_s/Ee

     # observed y
    noise = torch.randn(n)
    y = y_C * (f_y00 + alpha * f_t + beta * f_s + w_noise * noise * treatments)  # n
    y_0 = y_C * (f_y00 + 0 + beta * f_s)
    y_1 = y_C * (f_y00 + alpha * ites + beta * f_s + w_noise * noise)

    y = y * (1.0/(1+alpha+beta))
    y_0 = y_0 * (1.0/(1+alpha+beta))
    y_1 = y_1 * (1.0/(1+alpha+beta))

    y_0 = y_0.reshape(1, -1)
    y_1 = y_1.reshape(1, -1)
    Y_true = torch.cat([y_0, y_1], 0)

    print('noise:', torch.mean(w_noise * noise), torch.std(w_noise * noise))

    simulate_outcome_results = {'outcomes': y.cpu().numpy(), 'Y_true': Y_true.cpu().numpy()}
    return simulate_outcome_results

def simulate_outcome_quadratic(features, hyperedge_index, treatments, alpha=1.0, beta=1.0, nonlinear_type='sigmoid'):
    features.to(device)
    hyperedge_index = torch.tensor(hyperedge_index).type(torch.long)
    treatments = torch.tensor(treatments)

    n = features.shape[0]
    dim_size = features.shape[1]
    y_C = 0.3
    c_ft = 5
    w_noise = 1.0

    w0 = torch.randn(dim_size).to(device)
    f_y00 = torch.matmul(features, w0.reshape(-1)).to(device)

    w1 = torch.randn(dim_size, dim_size).to(device)
    w1 = w1.unsqueeze(0).repeat([n,1,1])
    ites = torch.bmm(features.unsqueeze(1), w1)
    ites.to(device)
    ites = torch.bmm(ites, features.unsqueeze(2)).squeeze() + c_ft
    f_t = ites * treatments

    tx = features * treatments.unsqueeze(1).repeat([1,dim_size])
    tx.to(device)
    f_tx = torch.bmm(tx.unsqueeze(1), w1).to(device)
    f_tx = torch.bmm(f_tx, tx.unsqueeze(2)).squeeze()
    f_t_he_ave = scatter_add(f_tx[hyperedge_index[0]], hyperedge_index[1], dim=0).to(device)
    Ne = scatter_add(torch.ones(hyperedge_index.size(1)), hyperedge_index[1], dim=0).to(device)
    f_t_he_ave = f_t_he_ave/(Ne**2)
    f_t_he_ave = non_linear(f_t_he_ave, nonlinear_type)
    f_s = scatter_add(f_t_he_ave[hyperedge_index[1]], hyperedge_index[0], dim=0).to(device)
    Ee = scatter_add(torch.ones(hyperedge_index.size(1)), hyperedge_index[0], dim=0).to(device)
    f_s = f_s/Ee

     # observed y
    noise = torch.randn(n)
    y = y_C * (f_y00 + alpha * f_t + beta * f_s + w_noise * noise * treatments)  # n
    y_0 = y_C * (f_y00 + 0 + beta * f_s)
    y_1 = y_C * (f_y00 + alpha * ites + beta * f_s + w_noise * noise)

    y = y * (1.0/(1+alpha+beta))
    y_0 = y_0 * (1.0/(1+alpha+beta))
    y_1 = y_1 * (1.0/(1+alpha+beta))

    y_0 = y_0.reshape(1, -1)
    y_1 = y_1.reshape(1, -1)
    Y_true = torch.cat([y_0, y_1], 0)

    print('noise:', torch.mean(w_noise * noise), torch.std(w_noise * noise))

    simulate_outcome_results = {'outcomes': y.cpu().numpy(), 'Y_true': Y_true.cpu().numpy()}
    return simulate_outcome_results

def load_data(result_path, path, exp_num=10):
    trn_rate = 0.6
    tst_rate = 0.2

    data = sio.loadmat(path)
    features, treatments, outcomes, Y_true, hyperedge_index = data['features'], data['treatments'][0], data['outcomes'][0], data['Y_true'], data['hyperedge_index']

    standarlize = True
    if standarlize:
        from sklearn import preprocessing
        scaler = preprocessing.StandardScaler().fit(features)
        features = scaler.transform(features)

    print('loaded data from ', path)

    show_hyperedge_size = True
    if show_hyperedge_size:
        unique, frequency = np.unique(hyperedge_index[1], return_counts=True)
        print('hyperedge size: ', np.sort(frequency)[::-1][:100])  # top 100 hyperedge size
        draw_freq(frequency, x_label='HyperEdges', bool_discrete=True, save_path = result_path+'Hyperedge.jpg')

    idx_trn_list, idx_val_list, idx_tst_list = [], [], []
    idx_treated = np.where(treatments == 1)[0]
    idx_control = np.where(treatments == 0)[0]
    for i in range(exp_num):
        idx_treated_cur = idx_treated.copy()
        idx_control_cur = idx_control.copy()
        np.random.shuffle(idx_treated_cur)
        np.random.shuffle(idx_control_cur)

        idx_treated_trn = idx_treated_cur[: int(len(idx_treated) * trn_rate)]
        idx_control_trn = idx_control_cur[: int(len(idx_control) * trn_rate)]
        idx_trn_cur = np.concatenate([idx_treated_trn, idx_control_trn])
        idx_trn_cur = np.sort(idx_trn_cur)
        idx_trn_list.append(idx_trn_cur)

        idx_treated_tst = idx_treated_cur[int(len(idx_treated) * trn_rate): int(len(idx_treated) * trn_rate) + int(len(idx_treated) * tst_rate)]
        idx_control_tst = idx_control_cur[int(len(idx_control) * trn_rate): int(len(idx_control) * trn_rate) + int(len(idx_control) * tst_rate)]
        idx_tst_cur = np.concatenate([idx_treated_tst, idx_control_tst])
        idx_tst_cur = np.sort(idx_tst_cur)
        idx_tst_list.append(idx_tst_cur)
        idx_treated_val = idx_treated_cur[int(len(idx_treated) * trn_rate) + int(len(idx_treated) * tst_rate):]
        idx_control_val = idx_control_cur[int(len(idx_control) * trn_rate) + int(len(idx_control) * tst_rate):]
        idx_val_cur = np.concatenate([idx_treated_val, idx_control_val])
        idx_val_cur = np.sort(idx_val_cur)
        idx_val_list.append(idx_val_cur)

    # tensor
    features = torch.FloatTensor(features)
    treatments = torch.FloatTensor(treatments)
    Y_true = torch.FloatTensor(Y_true)
    outcomes = torch.FloatTensor(outcomes)

    hyperedge_index = torch.LongTensor(hyperedge_index)
    idx_trn_list = [torch.LongTensor(id) for id in idx_trn_list]
    idx_val_list = [torch.LongTensor(id) for id in idx_val_list]
    idx_tst_list = [torch.LongTensor(id) for id in idx_tst_list]

    return features, treatments, outcomes, Y_true, hyperedge_index, idx_trn_list, idx_val_list, idx_tst_list