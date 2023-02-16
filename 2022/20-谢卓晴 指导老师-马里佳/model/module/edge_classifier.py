#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from model.module.biaffine import Biaffine


class EdgeClassifier(nn.Module):
    def __init__(self, dataset, args, initialize: bool, presence: bool, label: bool):
        super(EdgeClassifier, self).__init__()

        self.presence = presence
        if self.presence:
            if initialize:
                presence_init = torch.tensor([dataset.edge_presence_freq])
                presence_init = (presence_init / (1.0 - presence_init)).log()
            else:
                presence_init = None

            self.edge_presence = EdgeBiaffine(
                args.hidden_size, args.hidden_size_edge_presence, 1, args.dropout_edge_presence, bias_init=presence_init
            )

        self.label = label
        if self.label:
            label_init = (dataset.edge_label_freqs / (1.0 - dataset.edge_label_freqs)).log() if initialize else None
            n_labels = len(dataset.edge_label_field.vocab)
            self.edge_label = EdgeBiaffine(
                args.hidden_size, args.hidden_size_edge_label, n_labels, args.dropout_edge_label, bias_init=label_init
            )

    def forward(self, x):
        presence, label = None, None

        if self.presence:
            presence = self.edge_presence(x).squeeze(-1)  # shape: (B, T, T)
        if self.label:
            label = self.edge_label(x)  # shape: (B, T, T, O_1)

        return presence, label


class EdgeBiaffine(nn.Module):
    def __init__(self, hidden_dim, bottleneck_dim, output_dim, dropout, bias_init=None):
        super(EdgeBiaffine, self).__init__()
        self.hidden = nn.Linear(hidden_dim, 2 * bottleneck_dim)
        self.output = Biaffine(bottleneck_dim, output_dim, bias_init=bias_init)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.elu(self.hidden(x)))  # shape: (B, T, 2H)
        predecessors, current = x.chunk(2, dim=-1)  # shape: (B, T, H), (B, T, H)
        edge = self.output(current, predecessors)  # shape: (B, T, T, O)
        return edge
