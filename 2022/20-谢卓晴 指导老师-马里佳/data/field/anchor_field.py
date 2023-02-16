#!/usr/bin/env python3
# coding=utf-8

import torch
from data.field.mini_torchtext.field import RawField


class AnchorField(RawField):
    def process(self, batch, device=None):
        tensors, masks = self.pad(batch, device)
        return tensors, masks

    def pad(self, anchors, device):
        tensor = torch.zeros(anchors[0], anchors[1], dtype=torch.long, device=device)
        for anchor in anchors[-1]:
            tensor[anchor[0], anchor[1]] = 1
        mask = tensor.sum(-1) == 0

        return tensor, mask
