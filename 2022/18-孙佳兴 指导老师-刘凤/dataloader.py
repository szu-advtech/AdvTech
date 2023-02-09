import csv
import glob
import os

import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset


class DataSet(Dataset):

    def __init__(self, data_path, flag):

        super(DataSet, self).__init__()
        self.father_path = data_path
        self.flag = flag
        self.names = ['LRHS', 'PAN', 'gtHS']
        if flag == 'train':
            self.subpath = 'images_train.csv'
        else:
            self.subpath = 'images_test.csv'
        self.L, self.P, self.gt = self.get_data_paths()

    def get_data_paths(self):
        # csv文件不存在的话，读取mat文件存csv
        if not os.path.exists(os.path.join(self.father_path, self.subpath)):

            Ls = sorted(glob.glob(os.path.join(self.father_path, self.names[0], self.flag, '*.mat')),
                        key=lambda x: int(x.split('.')[0].split('_')[-1]))
            Ps = sorted(glob.glob(os.path.join(self.father_path, self.names[1], self.flag, '*.mat')),
                        key=lambda x: int(x.split('.')[0].split('_')[-1]))
            gts = sorted(glob.glob(os.path.join(self.father_path, self.names[2], self.flag, '*.mat')),
                         key=lambda x: int(x.split('.')[0].split('_')[-1]))

            with open(os.path.join(self.father_path, self.subpath), mode='w', newline='') as f:
                writer = csv.writer(f)
                for i in range(len(Ls)):
                    writer.writerow([Ls[i], Ps[i], gts[i]])

        L_list, P_list, gt_list = [], [], []
        with open(os.path.join(self.father_path, self.subpath)) as f:
            reader = csv.reader(f)
            for row in reader:
                l, p, g = row
                L_list.append(l)
                P_list.append(p)
                gt_list.append(g)

        assert len(L_list) == len(gt_list) and len(P_list) == len(gt_list)

        return L_list, P_list, gt_list

    def __len__(self):
        return len(self.L)

    def __getitem__(self, index):

        l, p, g = self.L[index], self.P[index], self.gt[index]

        LRHS = np.squeeze(sio.loadmat(l)['b'].astype(np.float32))
        PAN = sio.loadmat(p)['b'].astype(np.float32)
        PAN = np.reshape(PAN, [1, 160, 160])
        gtHS = np.squeeze(sio.loadmat(g)['b'].astype(np.float32))

        return LRHS, PAN, gtHS
