import torch
import numpy as np


def get_input(args, batch, _device, is_train=None):
    if is_train:
        item_seq = list(batch['item_seq'].values())
        user_seq = list(batch['uid'].values())
        behavior_seq = list(batch['behavior_seq'].values())
        len_seq = list(batch['len_seq'].values())
        target = list(batch['target'].values())
    else:
        item_seq = batch['init_item_seq'].values.tolist()
        user_seq = batch['uid'].values.tolist()
        behavior_seq = batch['init_behavior_seq'].values.tolist()
        len_seq = batch['len_seq'].values.tolist()
        target = batch['target'].values.tolist()

    pos_seq = np.zeros((len(item_seq), len(item_seq[0])))
    for i in range(len(item_seq)):
        for j in range(len(item_seq[0])):
            if behavior_seq[i][j] == 2:
                continue
            else:
                pos_seq[i][j] = j + 1

    item_seq, user_seq, behavior_seq, pos_seq = (torch.LongTensor(item_seq),
                                                           torch.LongTensor(user_seq),
                                                           torch.LongTensor(behavior_seq),
                                                           torch.LongTensor(pos_seq))

    item_seq, user_seq, behavior_seq, pos_seq, type_tag = (item_seq.to(_device),
                                                           user_seq.to(_device),
                                                           behavior_seq.to(_device),
                                                           pos_seq.to(_device))

    res = [item_seq, user_seq, behavior_seq, pos_seq, len_seq, target]
    return res
