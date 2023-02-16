#!/usr/bin/env python

import torch
import torch.nn as nn
from torch.nn.functional import dropout

from mrgcn.layers.graph import GraphConvolution
from mrgcn.data.batch import getAdjacencyNodeColumnIdx, A_Batch


class RGCN(nn.Module):
    def __init__(self, modules, num_relations, num_nodes, num_bases,
                 p_dropout, featureless, bias, link_prediction):
        """
        Relational Graph Convolutional Network

        PARAMETERS
        """
        super().__init__()

        assert len(modules) > 0

        self.num_nodes = num_nodes
        self.p_dropout = p_dropout

        self.layers = nn.ModuleDict()
        self.activations = nn.ModuleDict()
        # input layer
        indim, outdim, ltype, f_activation = modules[0]
        self.layers['layer_0'] = GraphConvolution(indim=indim,
                                          outdim=outdim,
                                          num_relations=num_relations,
                                          num_nodes=num_nodes,
                                          num_bases=num_bases,
                                          featureless=featureless,
                                          input_layer=True,
                                          bias=bias)
        self.activations['layer_0'] = f_activation

        # other layers (if any)
        for i, layer in enumerate(modules[1:], 1):
            indim, outdim, ltype, f_activation = layer
            self.layers['layer_'+str(i)] = GraphConvolution(indim=indim,
                                              outdim=outdim,
                                              num_relations=num_relations,
                                              num_nodes=num_nodes,
                                              num_bases=num_bases,
                                              featureless=False,
                                              input_layer=False,
                                              bias=bias)
            self.activations['layer_'+str(i)] = f_activation

        self.num_layers = len(self.layers)

        if link_prediction:
            # simulate diag(R) with R = (r x n x h) by vectors (r x h)
            size = (num_relations, modules[-1][1])
            self.relations = nn.Parameter(torch.empty(size))

            # initiate weights
            self.reset_parameters()

    def forward(self, X, A):
        if type(A) is A_Batch:
            return self._forward_mini_batch(X, A)

        return self._forward_full_batch(X, A)

    def _forward_full_batch(self, X, A):
        # Forward pass with full batch
        for layer, f_activation in zip(self.layers.values(),
                                       self.activations.values()):
            if type(layer) is GraphConvolution:
                X = layer(X, A)
            else:
                X = layer(X)

            if self.p_dropout > 0.0:
                # add dropout to output, by elementwise multiplying with 
                # column vector of ones, with dropout applied to the vector
                # of ones.
                ones = dropout(torch.ones(self.num_nodes),
                               p=self.p_dropout)
                X = torch.mul(X.T, ones).T

            if f_activation is not None:
                X = f_activation(X)

        return X

    def _forward_mini_batch(self, X, A):
        # Forward pass with mini batch
        for layer_idx, (layer, f_activation) in enumerate(zip(self.layers.values(),
                                                              self.activations.values())):

            if type(layer) is not GraphConvolution:
                X = layer(X)
                if f_activation is not None:
                    return f_activation(X)

            i = self.num_layers - (layer_idx + 1)  # most distant nodes
            A_slices= A.row[i]
            if layer.input_layer and layer.featureless:
                X = layer(None, A_slices)
            else:
                # compute embeddings of nodes i hops away, using
                # the embeddings of their neighbours at i+1 hops away.
                # use only the relevant subset of A, by omitting
                # irrelevant columns and rows.
                neighbours_idx = A.neighbours[i]
                A_idx = getAdjacencyNodeColumnIdx(neighbours_idx,
                                                   layer.num_nodes,
                                                   layer.num_relations)

                X = layer(X, A_slices, A_idx)

            if self.p_dropout > 0.0:
                # add dropout to output, by elementwise multiplying with 
                # column vector of ones, with dropout applied to the vector
                # of ones.
                ones = dropout(torch.ones(X.shape[0]),
                               p=self.p_dropout)
                X = torch.mul(X.T, ones).T

            if f_activation is not None:
                X = f_activation(X)
                
        return X

    def reset_parameters(self):
        # reset the edge embeddings when doing link prediction
        nn.init.xavier_uniform_(self.relations)
