#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.biaffine import Biaffine


class AnchorClassifier(nn.Module):
    def __init__(self, dataset, args, initialize: bool, bias=True, mode="anchor"):
        super(AnchorClassifier, self).__init__()

        self.token_f = nn.Linear(args.hidden_size, args.hidden_size_anchor)
        self.label_f = nn.Linear(args.hidden_size, args.hidden_size_anchor)
        self.dropout = nn.Dropout(args.dropout_anchor)

        if bias and initialize:
            bias_init = torch.tensor([getattr(dataset, f"{mode}_freq")])
            bias_init = (bias_init / (1.0 - bias_init)).log()
        else:
            bias_init = None

        self.output = Biaffine(args.hidden_size_anchor, 1, bias=bias, bias_init=bias_init)

    def forward(self, label, tokens, encoder_mask):
        tokens = self.dropout(F.elu(self.token_f(tokens)))  # shape: (B, T_w, H)
        label = self.dropout(F.elu(self.label_f(label)))  # shape: (B, T_l, H)
        anchor = self.output(label, tokens).squeeze(-1)  # shape: (B, T_l, T_w)

        anchor = anchor.masked_fill(encoder_mask.unsqueeze(1), float("-inf"))  # shape: (B, T_l, T_w)
        return anchor
