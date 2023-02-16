#!/usr/bin/env python

import numpy as np
import scipy.sparse as sp
import torch

from mrgcn.data.utils import (collate_padding,
                              collate_zero_padding_sparse,
                              scipy_sparse_to_pytorch_sparse)
from mrgcn.models.utils import getPadSymbol


class Batch:
    A = None
    X = None
    node_index = None
    device = None

    def __init__(self, batch_node_idx=None):
        self.device = torch.device('cpu')

        if batch_node_idx is not None:
            self.node_index = np.copy(batch_node_idx)

    def pad_(self, time_dim=1, pad_symbols=dict()):
        if self.X is None:
            return

        for i, (datatype, encoding_sets, _) in enumerate(self.X[1:], 1):
            for j, (encodings, _, seq_length) in enumerate(encoding_sets):
                if encodings.dtype != np.dtype("O"):  # object array
                    # no padding needed
                    continue

                 # pad time_dim such that the width of
                 # each matrix/array is as long as that
                 # of the widest member
                max_width = max(seq_length)
                sample = encodings[0]  # assume all members are of same type
                if isinstance(sample, np.ndarray):  # tokens
                    pad_symbol = 0
                    if datatype in pad_symbols.keys():
                        pad_symbol = pad_symbols[datatype]

                    encodings_padded = collate_padding(encodings,
                                                       pad_symbol=pad_symbol,
                                                       min_padded_length=max_width)
                else:
                    # sparse matrix
                    encodings_padded = collate_zero_padding_sparse(encodings,
                                                                   time_dim,
                                                                   min_padded_length=max_width)

                self.X[i][1][j][0] = encodings_padded

    def to_dense_(self):
        # make sparse CSR arrays dense
        if self.X is None:
            return

        for i, (_, encoding_sets, _) in enumerate(self.X[1:], 1):
            for j, (encodings, _, _) in enumerate(encoding_sets):
                if encodings.dtype == np.dtype("O"):
                    if not isinstance(encodings[0], sp.csr_matrix):
                        # assume members are of same type
                        continue

                    encodings_dense = np.array([a.todense() for a in encodings])

                    self.X[i][1][j][0] = encodings_dense

    def as_tensors_(self):
        self.node_index = torch.from_numpy(self.node_index)

        if self.X is None or isinstance(self.X[0], torch.Tensor):
            # if X[0] is a tensor, then so are its other members
            return

        self.X[0] = torch.from_numpy(self.X[0])
        for i, (_, encoding_sets, _) in enumerate(self.X[1:], 1):
            for j, (encodings, node_idx, seq_lengths) in enumerate(encoding_sets):
                self.X[i][1][j][0] = torch.from_numpy(encodings)
                self.X[i][1][j][1] = torch.from_numpy(node_idx)
                self.X[i][1][j][2] = torch.from_numpy(seq_lengths)
    
    #def to(self, device):
    #    device = torch.device(device) if type(device) is str else device

    #    copy = self
    #    if self.device is not device:
    #        copy = type(self)()
    #        copy.device = device

    #        copy.node_index = self.node_index.to(device)

    #        if self.X is None:
    #            return copy

    #        copy.X = list()
    #        copy.X.append(self.X[0].to(device))
    #        for datatype, encoding_sets, gpu_acceleration in self.X[1:]:
    #            copied_encoding_sets = list()
    #            for encodings, node_idx, seq_lengths in encoding_sets:
    #                copied_encoding_sets.append([encodings.to(device),
    #                                             node_idx.to(device),
    #                                             seq_lengths.to(device)])

    #            copy.X.append([datatype, copied_encoding_sets])
    #
    #    copy.A = self.A.to(device)

    #    return copy

    def to(self, devices):
        for i, (datatype, encoding_sets, _) in enumerate(self.X[1:], 1):
            device = devices[datatype]
            for j, (encodings, node_idx, seq_lengths) in enumerate(encoding_sets):
                self.X[i][1][j][0] = encodings.to(device)
                self.X[i][1][j][1] = node_idx.to(device)
                self.X[i][1][j][2] = seq_lengths.to(device)

        gcn_device = devices["relational"]
        self.A.to(gcn_device)

        device = "ambigious"
        devices = np.unique(devices)
        if len(devices) == 1:
            device = devices[0]

        self.device = device

        return self


class FullBatch(Batch):
    def __init__(self, A=None, X=None, batch_node_idx=None):
        super().__init__(batch_node_idx)

        if A is not None:
            self.A = A  # sparse COO tensor
        if X is not None:
            self.X = X  # X := [X, F1, F2, ..., Fn]

    def as_tensors_(self):
        super().as_tensors_()

        # this converts the sp CSR matrix to a COO tensor
        self.A = scipy_sparse_to_pytorch_sparse(self.A,
                                                dtype=torch.int8)
    

class MiniBatch(Batch):
    def __init__(self, A=None, X=None, batch_node_idx=None, num_layers=None):
        super().__init__(batch_node_idx)

        if A is not None:
            self.A = A_Batch(A, self.node_index, num_layers)

            if X is not None:  # if not featureless
                X_nodes = self.A.neighbours[-1]  # outer nodes
                self.X = mksubset(X, X_nodes)

    def as_tensors_(self):
        super().as_tensors_()
        self.A.as_tensors_()
    

class A_Batch:
    node_index = None  # nodes to compute embeddings for
    neighbours = None  # nodes necessary to compute embeddings
    row = None  # adjacency matrix slices necessary to compute embeddings
    device = None

    def __init__(self, A=None, batch_idx=None, num_layers=0):
        self.neighbours = list()
        self.row = list()

        self.device = torch.device('cpu')
        if batch_idx is not None:
            self.node_index = np.copy(batch_idx)

        if A is not None:
            self._populate(A, num_layers)

    def _populate(self, A, num_layers):
        sample_idx = self.node_index  # nodes to compute the embeddings for at layer i
        for _ in range(num_layers):       
            # slices of A belonging to the sample nodes at layer i
            A_samples = A[sample_idx]
            
            # neighbours of the sample nodes at layer i
            neighbours_idx = getNeighboursSparse(A, sample_idx)
                    
            self.row.append(A_samples)
            self.neighbours.append(neighbours_idx)
                
            sample_idx = neighbours_idx

    def to(self, device):
        self.node_index = self.node_index.to(device)
        self.neighbours = [t.to(device) for t in self.neighbours]
        self.row = [t.to(device) for t in self.row]

        self.device = device

        return self

    #def to(self, device):
    #    device = torch.device(device) if type(device) is str else device

    #    copy = self
    #    if self.device is not device:
    #        copy = A_Batch()

    #        copy.node_index = self.node_index.to(device)
    #        copy.neighbours = [t.to(device) for t in self.neighbours]
    #        copy.row = [t.to(device) for t in self.row]

    #        copy.device = device

    #    return copy
        
    def as_tensors_(self):
        self.node_index = torch.from_numpy(self.node_index)
        self.row = [scipy_sparse_to_pytorch_sparse(a, dtype=torch.int8) for a in self.row]
        self.neighbours = [torch.from_numpy(a) for a in self.neighbours]

def getNeighboursSparse(A, idx):
    """
    Return indices of neighbours of nodes with idx as indices, irrespective
    of relation.
    Assume A is a scypi CSR tensor
    """
    assert isinstance(A, sp.csr_matrix)

    # this operation is inefficient
    #neighbours_rel = [np.where(A[i].todense() == 1)[1] for i in idx]

    num_nodes = A.shape[0]
    neighbours_rel = [A.indices[A.indptr[i]:A.indptr[i+1]] for i in idx]
    neighbours_global = {i%num_nodes for i in np.concatenate(neighbours_rel)}

    return np.array(sorted(list(neighbours_global)))

def getAdjacencyNodeColumnIdx(idx, num_nodes, num_relations):
    """
    Return column idx for all nodes in idx for all relations
    """
    return torch.LongTensor([int((r*num_nodes) + i)
                             for r in range(num_relations) for i in idx])

def sliceSparseCOO(t, idx):
    assert isinstance(t, torch.sparse.CharTensor)
    row, col = t._indices()[:, torch.where(torch.isin(t._indices()[1],
                                                      idx))[0]]

    col_index_map = {int(j): i for i,j in enumerate(idx)}
    col = torch.LongTensor([col_index_map[int(i)] for i in col])
    
    return torch.sparse_coo_tensor(torch.vstack([row, col]),
                                                torch.ones(len(col),
                                                           dtype=torch.float32),
                                   size = [t.shape[0], len(idx)])

def mksubset(X, sample_idx):
    """ Subset data

        Create a subset of the raw input data for the sample nodes.
        Preserves the same structure
    """
    X, F = X[0], X[1:]

    X_sample = [X[sample_idx]]
    for modality, F_set, gpu_acceleration in F:
        # TODO: skip if modality not asked for (optimization)
        F_set_sample = list()
        for encodings, nodes_idx, seq_lengths in F_set:
            # find nodes in sample with this modality
            common_nodes = sorted(np.intersect1d(nodes_idx, sample_idx))
            num_common = len(common_nodes)
            if num_common <= 0:
                # add a dummy set if these encodings are not
                # available for this sample, to preserve order
                # in case different modules were created per
                # encoding set (not merged, not default)
                F_set_sample.append([np.empty(0),
                                     np.empty(0),
                                     np.empty(0)])

                continue

            # initiate subset structures
            if encodings.dtype == np.dtype("O"):  # object array
                encodings_sample = np.empty(shape=num_common,
                                            dtype=object)
            else:
                encodings_sample = np.zeros((num_common,
                                            *encodings.shape[1:]))

            nodes_idx_sample = np.array(common_nodes)
            seq_length_sample = np.zeros(num_common, dtype=int)

            # find indices for common nodes
            F_sample_mask = np.in1d(nodes_idx, common_nodes)

            seq_length_sample = seq_lengths[F_sample_mask]
            encodings_sample = encodings[F_sample_mask]

            F_set_sample.append([encodings_sample,
                                 nodes_idx_sample,
                                 seq_length_sample])

        X_sample.append([modality, F_set_sample, gpu_acceleration])

    return X_sample
