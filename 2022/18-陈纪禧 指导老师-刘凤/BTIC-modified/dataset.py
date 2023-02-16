import pandas as pd
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data.dataset import Dataset

image_path = '/data1/cjx/dataset/fakeddit/images'
data_path = './fakeddit'

text = pd.read_csv(f'{data_path}/all_data.tsv', sep='\t')

# print("loading image feature...")
# image = np.load(f'{data_path}/images/image_196_for_r50.npy', allow_pickle=True)
# print("finish loading")


def get_unique_index(id_str: str):
    # result_arr = np.char.find(image[:, 0].astype(str), id_str)
    # feature_idx = int(np.argwhere(result_arr == 0))
    id_line = text[text.id == id_str]
    unique_index = id_line.index[0]
    return unique_index


class SimDataset(Dataset):
    def __init__(self, mode: str):
        self.df = pd.read_csv(f"{data_path}/{mode}_new.tsv", sep='\t')

    def __getitem__(self, index):
        df_line = self.df.iloc[index]
        # 这里返回 image array 中的 index，image 中是三个表合并后的，所以有唯一的 index
        id_str = df_line["Id"]
        id_unique = get_unique_index(id_str)

        label = text.iloc[id_unique]["2_way_label"]

        i11 = get_unique_index(df_line["i11"])
        i12 = get_unique_index(df_line["i12"])
        i13 = get_unique_index(df_line["i13"])
        i14 = get_unique_index(df_line["i14"])
        i15 = get_unique_index(df_line["i15"])

        i01 = get_unique_index(df_line["i01"])
        i02 = get_unique_index(df_line["i02"])
        i03 = get_unique_index(df_line["i03"])
        i04 = get_unique_index(df_line["i04"])
        i05 = get_unique_index(df_line["i05"])

        return (torch.LongTensor([id_unique]),
                torch.LongTensor([label]),
                torch.LongTensor([i11]),
                torch.LongTensor([i12]),
                torch.LongTensor([i13]),
                torch.LongTensor([i14]),
                torch.LongTensor([i15]),
                torch.LongTensor([i01]),
                torch.LongTensor([i02]),
                torch.LongTensor([i03]),
                torch.LongTensor([i04]),
                torch.LongTensor([i05]),)

    def __len__(self):
        return len(self.df)
