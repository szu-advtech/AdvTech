#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.head.abstract_head import AbstractHead
from data.parser.to_mrp.sequential_parser import SequentialParser
from utility.cross_entropy import cross_entropy


class SequentialHead(AbstractHead):
    def __init__(self, dataset, args, initialize):
        config = {
            "label": True,
            "edge presence": False,
            "edge label": False,
            "anchor": True,
            "source_anchor": True,
            "target_anchor": True
        }
        super(SequentialHead, self).__init__(dataset, args, config, initialize)
        self.parser = SequentialParser(dataset)
