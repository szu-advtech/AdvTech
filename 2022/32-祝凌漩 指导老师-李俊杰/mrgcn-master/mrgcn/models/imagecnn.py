#!/usr/bin/env python

import torch
import torch.nn as nn

from mrgcn.models.utils import freeze_, inferOutputDim, stripClassifier


class ImageCNN(nn.Module):
    def __init__(self, model, output_dim, p_dropout=0.2,
                 bias=True, finetune=True):
        super().__init__()

        self.module_dict = nn.ModuleDict()

        self.finetune = finetune
        self.base_model = stripClassifier(model)
        if self.finetune:
            freeze_(self.base_model)
        self.module_dict['pretrained_head'] = self.base_model
        
        inter_dim = inferOutputDim(self.base_model)
        self.pre_fc = nn.Linear(inter_dim, inter_dim, bias=bias)
        self.fc = nn.Linear(inter_dim, output_dim, bias=bias)
        self.f_activation = nn.PReLU()
        self.module_dict['pre_fc'] = self.pre_fc
        self.module_dict['fc'] = self.fc
        self.module_dict['activation'] = self.f_activation

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.module_dict['pool'] = self.avgpool
        self.dropout = None
        if p_dropout > 0:
            self.dropout = nn.Dropout(p=p_dropout)
            self.module_dict['dropout'] = self.dropout

    def forward(self, X):
        output = self.base_model(X)
        output = self.avgpool(output)
        output = torch.flatten(output, 1)

        output = self.pre_fc(output)
        output = self.f_activation(output)
        if self.dropout is not None:
            output = self.dropout(output)

        return self.fc(output)
