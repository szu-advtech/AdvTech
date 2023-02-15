import numpy as np
import torch
import pandas as pd

# memory bank
def read_memory(Id, memory_bank, device):
    Idd = Id.cpu().numpy().squeeze()
    memory = []
    for i in range(len(Idd)):
        j = Idd[i]
        n = memory_bank[memory_bank.Id == j].shape[0]
        if n == 0:
            memory_ = torch.zeros([1, 24]).numpy().tolist()
            memory.append(memory_)
        else:
            memory_ = memory_bank[(memory_bank.Id == j)]
            memory__ = memory_['memory'].values.tolist()
            memory.append(memory__)
    memory = torch.tensor(memory).to(device)
    memory = memory.squeeze(1)
    return memory


def multiread(id1, id2, id3, id4, id5, memory_bank, device):
    x1 = read_memory(id1, memory_bank, device)
    x2 = read_memory(id2, memory_bank, device)
    x3 = read_memory(id3, memory_bank, device)
    x4 = read_memory(id4, memory_bank, device)
    x5 = read_memory(id5, memory_bank, device)
    return x1, x2, x3, x4, x5


def write_memory(Id, x, memory_bank):
    Idd = Id.cpu().numpy().squeeze()
    xx = x.detach().cpu().numpy().squeeze()
    for i in range(len(Idd)):
        j = Idd[i]
        a = xx[i]
        n = memory_bank[memory_bank.Id == j].shape[0]
        if n == 0:
            row = {'Id': [j], 'memory': [a]}
            memory_bank = pd.concat([memory_bank, pd.DataFrame(row)])
        else:
            b = memory_bank.loc[memory_bank['Id'] == j].index.tolist()
            memory_bank = memory_bank.drop(b)
            row = {'Id': [j], 'memory': [a]}
            memory_bank = pd.concat([memory_bank, pd.DataFrame(row)])
    return memory_bank


def multisim(x, x1, x2, x3, x4, x5):
    s1 = (torch.cosine_similarity(x, x1, dim=1) + 1) / 2
    s2 = (torch.cosine_similarity(x, x2, dim=1) + 1) / 2
    s3 = (torch.cosine_similarity(x, x3, dim=1) + 1) / 2
    s4 = (torch.cosine_similarity(x, x4, dim=1) + 1) / 2
    s5 = (torch.cosine_similarity(x, x5, dim=1) + 1) / 2
    sim = (s1 + s2 + s3 + s4 + s5) / 5
    return sim


def consim(x, id1, id2, id3, id4, id5, memory_bank, device):
    x1, x2, x3, x4, x5 = multiread(id1, id2, id3, id4, id5, memory_bank, device)
    con_sim = multisim(x, x1, x2, x3, x4, x5)
    return con_sim


def flat_accuracy(preds, labels):
    """
    由 predictions 和 labels 计算 acc
    """
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

