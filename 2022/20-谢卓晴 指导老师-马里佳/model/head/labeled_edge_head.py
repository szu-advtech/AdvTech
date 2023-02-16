#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn as nn

from model.head.abstract_head import AbstractHead
from data.parser.to_mrp.labeled_edge_parser import LabeledEdgeParser
from utility.cross_entropy import binary_cross_entropy
from utility.hungarian_matching import match_label


class LabeledEdgeHead(AbstractHead):
    def __init__(self, dataset, args, initialize):
        config = {
            "label": True,
            "edge presence": True,
            "edge label": True,
            "anchor": True,
            "source_anchor": False,
            "target_anchor": False
        }
        super(LabeledEdgeHead, self).__init__(dataset, args, config, initialize)

        self.top_node = nn.Parameter(torch.randn(1, 1, args.hidden_size), requires_grad=True)
        self.parser = LabeledEdgeParser(dataset)

    def init_label_classifier(self, dataset, args, config, initialize: bool):
        classifier = nn.Sequential(
            nn.Dropout(args.dropout_label),
            nn.Linear(args.hidden_size, 1, bias=True)
        )
        if initialize:
            bias_init = torch.tensor([dataset.label_freqs[1]])
            classifier[1].bias.data = (bias_init / (1.0 - bias_init)).log()

        return classifier

    def forward_label(self, decoder_output):
        return self.label_classifier(decoder_output)

    def forward_edge(self, decoder_output):
        top_node = self.top_node.expand(decoder_output.size(0), -1, -1)
        decoder_output = torch.cat([top_node, decoder_output], dim=1)
        return self.edge_classifier(decoder_output)

    def loss_label(self, prediction, target, mask, matching):
        prediction = prediction["label"]
        target = match_label(
            target["labels"][0], matching, prediction.shape[:-1], prediction.device, self.query_length
        )
        return {"label": binary_cross_entropy(prediction.squeeze(-1), target.float(), mask, focal=self.focal)}

    def inference_label(self, prediction):
        return (prediction.squeeze(-1) > 0.0).long()

    def label_cost_matrix(self, output, batch, decoder_lens, b: int):
        if output["label"] is None:
            return 1.0

        target_labels = batch["anchored_labels"][b]  # shape: (num_nodes, num_inputs, 2)
        label_prob = output["label"][b, : decoder_lens[b], :].sigmoid().unsqueeze(0)  # shape: (1, num_queries, 1)
        label_prob = torch.cat([1.0 - label_prob, label_prob], dim=-1)  # shape: (1, num_queries, 2)
        tgt_label = target_labels.repeat_interleave(self.query_length, dim=1)  # shape: (num_nodes, num_queries, 2)
        cost_matrix = ((tgt_label * label_prob).sum(-1) * label_prob[:, :, 1:].sum(-1)).t().sqrt()  # shape: (num_queries, num_nodes)

        return cost_matrix
