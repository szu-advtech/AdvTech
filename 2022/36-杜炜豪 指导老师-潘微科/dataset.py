import math
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse


def loadTargetData(args):
    file = args.path + '/' + args.dataset + '/' + args.transaction
    tp = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    tp = tp.sort_values("uid")
    usersNum, itemsNum = args.user_num + 1, args.item_num + 1
    targetDict = tp.groupby('uid')['iid'].apply(list).to_dict()

    rows, cols = tp['uid'], tp['iid']
    targetData = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64', shape=(usersNum, itemsNum))
    return targetData, targetDict, usersNum, itemsNum


def loadAuxiliaryData(args):
    file = args.path + '/' + args.dataset + '/' + args.examination
    tp = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    tp = tp.sort_values("uid")
    usersNum, itemsNum = args.user_num + 1, args.item_num + 1
    auxiliaryDict = tp.groupby('uid')['iid'].apply(list).to_dict()

    rows, cols = tp['uid'], tp['iid']
    auxiliaryData = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64', shape=(usersNum, itemsNum))
    return auxiliaryData, auxiliaryDict


def loadTestData(args):
    file = args.path + '/' + args.dataset + '/' + args.test
    tp = pd.read_csv(file, sep=' ', names=['uid', 'iid'])
    tp = tp.sort_values("uid")
    testDict = tp.groupby('uid')['iid'].apply(list).to_dict()
    return testDict

#load the neighbor data
def loadNeighborData(args):
    file = args.path + '/' + args.dataset + '/neighbor_TA_pair.txt'
    tp = pd.read_csv(file, sep=' ', names=['uid', 'nbid'])
    tp = tp.sort_values("uid")
    usersNum= args.user_num + 1
    neighborDict = tp.groupby('uid')['nbid'].apply(list).to_dict()

    rows, cols = tp['uid'], tp['nbid']
    neighborData = sparse.csr_matrix(
        (np.ones_like(rows), (rows, cols)), dtype='float64', shape=(usersNum, usersNum))
    return neighborData, neighborDict

#process the neighbor data to pair
def transfer2pair():
    f1 = open('neighbor_TA.txt')
    f2 = open('mask_TA.txt')
    neighbor = f1.readlines()
    mask = f2.readlines()

    filename = 'neighbor_TA_pair.txt'
    with open(filename, 'a') as file_obj:
        for i, record in enumerate(neighbor):
            tmp_mask = mask[i].split(' ')
            tmp_num = int(sum(list(map(float, tmp_mask))))
            nb_list = record.split(' ')
            nb_list = list(map(int, nb_list))[:tmp_num]
            for nb in nb_list:
                file_obj.write(str(i)+' '+str(nb)+'\n')
    f1.close()
    f2.close()