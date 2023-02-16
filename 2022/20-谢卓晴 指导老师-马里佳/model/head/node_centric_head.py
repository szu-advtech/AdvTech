#!/usr/bin/env python3
# coding=utf-8

import torch

from model.head.abstract_head import AbstractHead
from data.parser.to_mrp.node_centric_parser import NodeCentricParser
from utility.cross_entropy import binary_cross_entropy


class NodeCentricHead(AbstractHead):
    def __init__(self, dataset, args, initialize):
        config = {
            "label": True,
            "edge presence": True,
            "edge label": False,
            "anchor": True,
            "source_anchor": False,
            "target_anchor": False
        }
        super(NodeCentricHead, self).__init__(dataset, args, config, initialize)

        self.source_id = dataset.label_field.vocab.stoi["Source"] + 1
        self.target_id = dataset.label_field.vocab.stoi["Target"] + 1
        self.parser = NodeCentricParser(dataset)
