# -*- coding = utf-8 -*-
# @Time : 2022-01-17 16:26
# @Author : XiaoJing
# @File : generate_input.py
# @Software : PyCharm

import torch


def get_input(batch, _device, is_train=None):
    if is_train:
        item_seq = list(batch['item_seq'].values())
        behavior_seq = list(batch['behavior_seq'].values())
        len_seq = list(batch['len_seq'].values())
        target = list(batch['target'].values())
    else:
        item_seq = batch['init_item_seq'].values.tolist()
        behavior_seq = batch['init_behavior_seq'].values.tolist()
        len_seq = batch['len_seq'].values.tolist()
        target = batch['target'].values.tolist()

    item_seq, behavior_seq = (torch.LongTensor(item_seq), torch.LongTensor(behavior_seq))

    item_seq, behavior_seq = (item_seq.to(_device), behavior_seq.to(_device))
    res = [item_seq, behavior_seq, len_seq, target]
    return res
