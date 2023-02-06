#!/usr/bin/env python

import logging
import os
from os import access, F_OK, R_OK, W_OK
from os.path import split
import random

import numpy as np
from rdflib.term import URIRef
import torch
from torch.utils.data import Dataset
import torch.nn.functional as f
import scipy.sparse as sp


logger = logging.getLogger(__name__)

def getConfParam(config, name, default=None):
    """ Access nested dict by dot-separated key strings
    """
    names = name.split('.')
    for name in names:
        if not isinstance(config, dict) or name not in config.keys():
            return default

        config = config[name]

    return config

def is_readable(filename):
    path = split(filename)[0]
    if not access(path, F_OK):
        raise OSError(":: Path does not exist: {}".format(path))
    elif not access(path, R_OK):
        raise OSError(":: Path not readable by user: {}".format(path))

    return True

def is_writable(filename):
    path = split(filename)[0]
    if not access(path, F_OK):
        raise OSError(":: Path does not exist: {}".format(path))
    elif not access(path, W_OK):
        raise OSError(":: Path not writeable by user: {}".format(path))

    return True

def is_gzip(filename):
    return True if filename.endswith('.gz') else False

def set_seed(seed=-1):
    if seed < 0:
        seed = np.random.randint(0, 2**32-1)

    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    logger.debug("Setting seed to {}".format(seed))

def strip_graph(knowledge_graph, config):
    target_property_inv = config['task']['target_property_inv']
    if target_property_inv == '':
        return

    n = len(knowledge_graph)
    separate_literals = config['graph']['structural']['separate_literals']
    logger.debug("Stripping knowledge graph...")
    # remove inverse target relations to prevent information leakage
    inv_target_triples = frozenset(knowledge_graph.triples((None,
                                                      URIRef(target_property_inv),
                                                      None),
                                                    separate_literals))
    knowledge_graph.graph -= inv_target_triples

    m = len(knowledge_graph)
    logger.debug("stripped {} triples ({} remain)".format(n-m, m))

def dataset_to_device(dataset, device):
    for split in dataset.values():
        split['Y'] = split['Y'].to(device)
        split['idx'] = split['idx'].to(device)
        # X stays where it is

def triples_to_indices(kg, node_map, edge_map, separate_literals=False):
    data = np.zeros((len(kg), 3), dtype=np.int32)
    for i, (s, p, o) in enumerate(kg.triples(separate_literals=separate_literals)):
        data[i] = np.array([node_map[s], edge_map[p], node_map[o]], dtype=np.int32)

    return data

class SparseDataset(Dataset):
    n = 0

    def __init__(self, sp_input):
        # sp_input := a list with sparse coo matrices
        self.n = len(sp_input)
        self.sp = sp_input

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.as_tensor(self.sp[idx].todense())

def collate_zero_padding_sparse(batch, time_dim, max_batch_length=999,
                                min_padded_length=5):
    """ batch := an object array with sparse csr matrices

        time_dim should be 0 for RNN and 1 for temporal CNN
        min_padded_length >= 5 to allow smallest CNN to support it
    """
    n = len(batch)
    batch_padded = np.empty(shape=n, dtype=object)

    max_length = 0
    for seq in batch:
        if seq.shape[time_dim] > max_length:
            max_length = seq.shape[time_dim]
    max_length = min(max_length, max_batch_length)
    padded_length = max(min_padded_length, max_length)

    for i, seq in enumerate(batch):
        shape = (seq.shape[0], padded_length) if time_dim == 1\
                else (padded_length, seq.shape[1])

        a = sp.csr_matrix((seq.data, seq.indices, seq.indptr),
                          shape=shape, dtype=seq.dtype)
        batch_padded[i] = a

    return batch_padded

def collate_padding(batch, pad_symbol=0, max_batch_length=999,
                         min_padded_length=5):
    """ batch := an object array with ndarrays of variable length
    """
    n = len(batch)

    max_length = np.max([len(a) for a in batch])
    max_length = min(max_length, max_batch_length)
    padded_length = max(min_padded_length, max_length)

    batch_padded = -np.ones(shape=(n, padded_length), dtype=int)
    for i, seq in enumerate(batch):
        batch_padded[i,:len(seq)] = seq
    
    batch_padded[batch_padded==-1] = pad_symbol

    return batch_padded


def zero_pad(t, min_seq_length, time_dim):
    if time_dim < 0 or min_seq_length < 0:
        return t

    dim = 1 if time_dim == 1 else 3
    padding = [0, 0, 0, 0]
    padding[dim] = min_seq_length - t.shape[time_dim]

    return f.pad(t, padding)

def scipy_sparse_to_pytorch_sparse(sp_input, dtype):
    indices = np.array(sp_input.nonzero())
    return torch.sparse_coo_tensor(torch.LongTensor(indices),
                                   torch.Tensor(sp_input.data),
                                   sp_input.shape,
                                   dtype=dtype)

