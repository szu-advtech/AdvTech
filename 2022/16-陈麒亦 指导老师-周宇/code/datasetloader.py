import csv

import numpy as np
import torch
import random
import train_process

def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cd_data = []
        cd_data += [[float(i) for i in row] for row in reader]
        return torch.Tensor(cd_data)


def get_edges(matrix):
    edges=[[],[]]
    for i in range(matrix.size(0)):
        for j in range(matrix.size(1)):
            if matrix[i][j] != 0:
                edges[0].append(i)
                edges[1].append(j)
    return torch.LongTensor(edges)


def get_GCN_dataset(cdpath,ccpath,ddpath):

    cd_matrix=read_csv(cdpath)
    zero_index = [[i, j, 0] for i in range(cd_matrix.size(0)) for j in range(cd_matrix.size(1)) if cd_matrix[i][j] < 1 ]
    one_index = [[i, j, 1] for i in range(cd_matrix.size(0)) for j in range(cd_matrix.size(1)) if cd_matrix[i][j] >= 1 ]

    #
    cd_pairs = random.sample(zero_index, len(one_index)) + one_index

    dd_matrix = read_csv(ccpath)
    dd_edges = get_edges(dd_matrix)


    cc_matrix = read_csv(ddpath)
    cc_edges = get_edges(cc_matrix)

    return {"dd_matrix":dd_matrix,"dd_edges":dd_edges,"cc_matrix":cc_matrix,"cc_edges":cc_edges,"cd_pairs":cd_pairs}

#
def get_cdmatrix(cd_pairs,train_index,test_index):
    cd_matrix = np.zeros((585, 88))
    for i in train_index:
        if cd_pairs[i][2] == 1:
            cd_matrix[cd_pairs[i][0]][cd_pairs[i][1]] = 1

    cd_matrix = torch.Tensor(cd_matrix)
    train_cd_pairs = []
    test_cd_pairs = []

    for m in train_index:
        train_cd_pairs.append(cd_pairs[m])

    for n in test_index:
        test_cd_pairs.append(cd_pairs[n])

    return cd_matrix, train_cd_pairs, test_cd_pairs

def feature_representation(model, dataset,lr=0.003,epoch=400):
    if torch.cuda.is_available():
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr)
    model = train_process.train(model, dataset, optimizer,epoch)
    model.eval()
    with torch.no_grad():
        score, cir_fea, dis_fea = model(dataset)
    cir_fea = cir_fea.cpu().detach().numpy()
    dis_fea = dis_fea.cpu().detach().numpy()
    return score, cir_fea, dis_fea

def get_clf_dataset(cir_fea, dis_fea, cd_pairs):
    unknown_pairs = []
    known_pairs = []

    for pair in cd_pairs:
        if pair[2] == 1:
            known_pairs.append(pair[:2])
        if pair[2] == 0:
            unknown_pairs.append(pair[:2])



    nega_list = []
    for i in range(len(unknown_pairs)):
        nega = cir_fea[unknown_pairs[i][0], :].tolist() + dis_fea[unknown_pairs[i][1], :].tolist() + [0, 1]
        nega_list.append(nega)

    posi_list = []
    for j in range(len(known_pairs)):
        posi = cir_fea[known_pairs[j][0], :].tolist() + dis_fea[known_pairs[j][1], :].tolist() + [1, 0]
        posi_list.append(posi)

    samples = posi_list + nega_list

    random.shuffle(samples)
    samples = np.array(samples)
    return samples


if __name__ == '__main__':

     print(len(get_GCN_dataset()['cd_pairs']))

