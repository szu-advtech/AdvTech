#!/usr/bin/env python

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self,
                 input_dim,
                 output_dim,
                 num_layers=3,
                 p_dropout=0.0,
                 bias=True):
        """
        Multi-Layer Perceptron with N layers

        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.p_dropout = p_dropout
        step_size = (input_dim-output_dim)//num_layers
        hidden_dims = [output_dim + (i * step_size)
                       for i in reversed(range(num_layers))]

        mlp = list()
        layer_indim = input_dim
        for hidden_dim in hidden_dims:
            mlp.extend([nn.Linear(layer_indim, hidden_dim, bias),
                        nn.Dropout(p=self.p_dropout, inplace=True),
                        nn.PReLU()])

            layer_indim = hidden_dim

        self.mlp = nn.Sequential(*mlp)

        # initiate weights
        self.init()

    def forward(self, X):
        return self.mlp(X)

    def init(self):
        for param in self.parameters():
            nn.init.uniform_(param)
