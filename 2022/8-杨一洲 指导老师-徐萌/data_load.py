import re

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data
import os


class My1x1Dataset(Data.Dataset):
    def __init__(self, start_year, end_year, attr):
        self.attr = attr
        root_path = f'/home3/yyz_/dataset/1x1/dataset/{attr}'
        self.file_path_list = []
        self.start_year = start_year
        self.end_year = end_year
        print('loading dataset')
        for year in range(start_year, end_year + 1):
            for month in range(1, 2):
                print(year, month)
                s_path = os.path.join(root_path, str(year), str(month))
                profile_list_in_month = os.listdir(s_path)
                # 过滤出feature
                x_file_name_list = list(filter(lambda x: re.match('.*?-x.ds', x), profile_list_in_month))
                # 拼接完整出完整路径
                file_path_list = list(map(lambda x: s_path + '/' + x, x_file_name_list))
                self.file_path_list += file_path_list

        self.number = len(self.file_path_list)

    def __len__(self):
        return self.number

    def __getitem__(self, index):
        x_path = self.file_path_list[index]
        y_path = x_path.replace('-x.ds', f'-{self.attr}_y.ds')
        x = torch.load(x_path)
        y = torch.load(y_path)
        return x, y


if __name__ == '__main__':
    m = My1x1Dataset(2012, 2015, 'temp')
    x, y = m.__getitem__(0)
    print(x.shape)
    print(y.shape)
    print(1)

