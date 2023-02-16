'''
Generate processed data: filtering, combining ...
'''

import numpy as np
import pickle
from util import hypergraph_stats, simulate_outcome_linear, simulate_outcome_quadratic, non_linear
import torch
import scipy.io as sio

# 生成超图
def preprocess_contact():
    path_nverts = 'contact-high-school-nverts.txt'
    path_simplices = 'contact-high-school-simplices.txt'

    # size of each hyperedge
    with open(path_nverts) as f:
        sizeOfEdge = f.readlines()
    f.close()
    sizeOfEdge = [int(i) for i in sizeOfEdge]
    m = len(sizeOfEdge)

    idx_start = []
    sum_size = 0
    for i in range(m):
        idx_start.append(sum_size)
        sum_size += sizeOfEdge[i]

    # nodes in each hyperedge
    with open(path_simplices) as f:
        edge_idx_node = f.readlines()
    f.close()
    edge_idx_node = [int(i) for i in edge_idx_node]
    # edge_idx_edge = [i for i in range(m) for j in range(sizeOfEdge[i])]

    # remove redundant hyperedges
    unique_edges = {}
    edge_idx_node_unique = []
    edge_idx_edge_unique = []
    for i in range(m):
        key_nodes = edge_idx_node[idx_start[i]: idx_start[i] + sizeOfEdge[i]]
        key_nodes.sort()
        key = ''
        for k in key_nodes:
            key += '_'+str(k)
        if key not in unique_edges:
            edge_idx_edge_unique += [len(unique_edges) for j in range(sizeOfEdge[i])]
            edge_idx_node_unique += edge_idx_node[idx_start[i]: idx_start[i] + sizeOfEdge[i]]
            unique_edges[key] = 1

    edge_idx_node_unique = [i-1 for i in edge_idx_node_unique]  # start from 0
    hyperedge_index = np.array([edge_idx_node_unique, edge_idx_edge_unique])

    # statistics
    n = np.max(hyperedge_index[0]) + 1
    statistics = hypergraph_stats(hyperedge_index, n)
    print(statistics)

    # record_data
    data_save = {'hyper_index': hyperedge_index}

    save_flag = True
    if save_flag:
        with open('contact_hypergraph.pickle', 'wb') as f:
            pickle.dump(data_save, f)
    return data_save


def simulation(type='all', alpha=1, beta=1, nonlinear_type='raw'):
    with open('contact_hypergraph.pickle', 'rb') as f:
        data = pickle.load(f)
    hyperedge_index = data['hyper_index']
    n = np.max(hyperedge_index[0]) + 1

    # simulate features
    d_x = 50
    features = torch.randn(n,d_x)
    print('feature std: ', torch.mean(torch.std(features, dim=0)))

    # simulate treatments
    W = torch.randn(d_x).reshape(-1,1)
    treatment_orin = non_linear(0.01 * torch.matmul(features, W), 'sigmoid')
    treatment_orin = treatment_orin.reshape(-1)
    treated_ratio = 0.49
    thresh_t = np.sort(treatment_orin)[::-1][int(treated_ratio * len(treatment_orin))]
    treatment = np.zeros_like(treatment_orin)
    treatment[np.where(treatment_orin >= thresh_t)] = 1.0
    treatment[np.where(treatment_orin < thresh_t)] = 0.0
    treated_ratio = float(np.count_nonzero(treatment)) / n
    print('treatment ratio: ', treated_ratio)

    # simulate outcomes
    if type != 'quadratic':
        simulate_outcome_linear_results = simulate_outcome_linear(features, hyperedge_index, treatment, alpha, beta, nonlinear_type)
        simulation_data = {
            'parameter': {'alpha': alpha, 'beta': beta, 'type': type, 'nonlinear_type': nonlinear_type},
            'features': features.cpu().numpy(),
            'treatments': treatment,
            'outcomes': simulate_outcome_linear_results['outcomes'],
            'Y_true': simulate_outcome_linear_results['Y_true'],
            'hyperedge_index': hyperedge_index
        }
        path_save = '../simulation/contact/contact_linear_alpha' + str(alpha) + '_beta' + str(beta) + '_' + nonlinear_type + '.mat'
        sio.savemat(path_save, simulation_data)
        print('Data saved! Path: ', path_save)
        print('type=', type, ' alpha=', alpha, ' beta=', beta)

    if type != 'linear':
        simulate_outcome_quadratic_results = simulate_outcome_quadratic(features, hyperedge_index, treatment, alpha, beta,nonlinear_type)
        simulation_data = {
            'parameter': {'alpha': alpha, 'beta': beta, 'type': type, 'nonlinear_type': nonlinear_type},
            'features': features.cpu().numpy(),
            'treatments': treatment,
            'outcomes': simulate_outcome_quadratic_results['outcomes'],
            'Y_true': simulate_outcome_quadratic_results['Y_true'],
            'hyperedge_index': hyperedge_index
        }
        path_save = '../simulation/contact/contact_quadratic_alpha' + str(alpha) + '_beta' + str(beta) + '_' + nonlinear_type + '.mat'
        sio.savemat(path_save, simulation_data)
        print('Data saved! Path: ', path_save)
        print('type=', type, ' alpha=', alpha, ' beta=', beta)



if __name__ == '__main__':
    # preprocess_contact()
    simulation()






