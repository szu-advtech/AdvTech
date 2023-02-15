#!/usr/bin/env python3
# coding=utf-8

import torch.nn as nn
from model.module.bilinear import Bilinear


class Biaffine(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, bias_init=None):
        super(Biaffine, self).__init__()

        self.linear_1 = nn.Linear(input_dim, output_dim, bias=False)
        self.linear_2 = nn.Linear(input_dim, output_dim, bias=False)

        self.bilinear = Bilinear(input_dim, input_dim, output_dim, bias=bias)
        if bias_init is not None:
            self.bilinear.bias.data = bias_init

    def forward(self, x, y):
        return self.bilinear(x, y) + self.linear_1(x).unsqueeze(2) + self.linear_2(y).unsqueeze(1)
