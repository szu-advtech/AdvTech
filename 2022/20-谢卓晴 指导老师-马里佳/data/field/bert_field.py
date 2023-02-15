#!/usr/bin/env python3
# coding=utf-8

import torch
from data.field.mini_torchtext.field import RawField


class BertField(RawField):
    def __init__(self):
        super(BertField, self).__init__()

    def process(self, example, device=None):
        attention_mask = [1] * len(example)

        example = torch.LongTensor(example, device=device)
        attention_mask = torch.ones_like(example)

        return example, attention_mask
