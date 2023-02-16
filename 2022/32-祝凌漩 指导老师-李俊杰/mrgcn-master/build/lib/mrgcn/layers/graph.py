#!/usr/bin/env python

from mrgcn.data.batch import sliceSparseCOO
import torch
import torch.nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, indim, outdim, num_relations, num_nodes, num_bases=-1,
                 bias=False, input_layer=False, featureless=False,
                 shared_bases_weights=False):
        """
        Relational Graph Convolutional Layer
        Mini batch support
        """
        super().__init__()

        self.indim = indim  # no. of input features
        self.outdim = outdim  # no. of output features
        self.num_relations = num_relations  # no. of relations
        self.num_nodes = num_nodes  # no. of nodes (batch size)
        self.num_bases = num_bases
        self.input_layer = input_layer
        self.featureless = featureless
        self.bias = bias

        self.weight_I = None
        self.weight_F = None
        self.weight_I_comp = None
        self.weight_F_comp = None
        self.b = None

        S = self.num_relations
        if self.num_bases > 0:
            # weights for bases matrices for identities and features
            S = self.num_bases

            if self.input_layer:
                self.weight_I_comp = nn.Parameter(torch.empty((self.num_relations,
                                                               self.num_bases)))
            if not self.featureless:
                if shared_bases_weights:
                    # use same basis matrix for both identities and features
                    self.weight_F_comp = self.weight_I_comp
                else:
                    self.weight_F_comp = nn.Parameter(torch.empty((self.num_relations,
                                                                   self.num_bases)))

        # weights for identities and features
        if self.input_layer:
            self.weight_I = nn.Parameter(torch.empty((S*self.num_nodes, self.outdim)))
        if not self.featureless:
            self.weight_F = nn.Parameter(torch.empty((S, self.indim, self.outdim)))

        # declare bias
        if self.bias:
            self.b = nn.Parameter(torch.empty(self.outdim))

        # initialize weights
        self.reset_parameters()

    def forward(self, X, A, A_idx=None):
        # if input layer: AXW = A[I F]W = AIW_I + AFW_F
        # else:           AXW = AHW

        AIW_I = 0.0
        if self.input_layer:
            W_I = self.weight_I
            if self.num_bases > 0:
                W_I = W_I.view(self.num_bases, self.num_nodes, self.outdim)
                W_I = torch.einsum('rb,bij->rij', self.weight_I_comp, W_I)
                W_I = W_I.view(self.num_relations*self.num_nodes, self.outdim)

            # AIW_I = AW_I
            AIW_I = torch.mm(A.float(), W_I)

            if self.featureless:
                if self.bias:
                    AIW_I = torch.add(AIW_I, self.b)

                return AIW_I

        W_F = self.weight_F
        if self.num_bases > 0:
            W_F = torch.einsum('rb,bij->rij', self.weight_F_comp, W_F)

        num_nodes = self.num_nodes
        if A_idx is not None:
            # mini batch mode
            num_nodes = X.shape[0]  # num nodes in batch
            A = sliceSparseCOO(A, A_idx)  # slice sparse COO tensor

        FW_F = torch.einsum('ij,bjk->bik', X, W_F)
        FW_F = torch.reshape(FW_F, (self.num_relations*num_nodes, self.outdim))
        AFW_F = torch.mm(A.float(), FW_F)

        AXW = torch.add(AIW_I, AFW_F) if self.input_layer else AFW_F

        if self.bias:
            AXW = torch.add(AXW, self.b)

        return AXW

    def reset_parameters(self):
        # initialize weights from a uniform distribution following
        # "Understanding the difficulty of training deep feedforward
        #  neural networks" - Glorot, X. & Bengio, Y. (2010)
        for name, param in self.named_parameters():
            if name == 'b':
                # skip bias
                continue
            nn.init.xavier_uniform_(param)

        # initialize bias as null vector
        if self.bias:
            nn.init.zeros_(self.b)
