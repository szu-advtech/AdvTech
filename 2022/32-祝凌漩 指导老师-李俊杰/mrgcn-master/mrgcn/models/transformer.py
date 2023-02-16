#!/usr/bin/env python

import torch.nn as nn

from mrgcn.models.utils import freeze_, inferOutputDim


class Transformer(nn.Module):
    def __init__(self, model, output_dim, p_dropout=0.2,
                 bias=True, finetune=True):
        super().__init__()

        self.module_dict = nn.ModuleDict()

        self.finetune = finetune
        self.base_model = model
        if self.finetune:
            freeze_(self.base_model)
        self.module_dict['pretrained_head'] = self.base_model
        
        inter_dim = inferOutputDim(model)
        self.pre_fc = nn.Linear(inter_dim, inter_dim, bias=bias)
        self.fc = nn.Linear(inter_dim, output_dim, bias=bias)
        self.f_activation = nn.PReLU()
        self.module_dict['pre_fc'] = self.pre_fc
        self.module_dict['fc'] = self.fc
        self.module_dict['activation'] = self.f_activation

        self.dropout = None
        if p_dropout > 0:
            self.dropout = nn.Dropout(p=p_dropout)
            self.module_dict['dropout'] = self.dropout

    def forward(self, X):
        hidden_state = self.base_model(X)[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_fc(pooled_output)
        pooled_output = self.f_activation(pooled_output)

        if self.dropout is not None:
            pooled_output = self.dropout(pooled_output)

        return self.fc(pooled_output)
