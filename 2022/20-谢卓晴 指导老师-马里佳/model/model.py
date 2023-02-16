#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn as nn

from model.module.encoder import Encoder

from model.module.transformer import Decoder
from model.head.node_centric_head import NodeCentricHead
from model.head.labeled_edge_head import LabeledEdgeHead
from model.head.sequential_head import SequentialHead
from utility.utils import create_padding_mask


class Model(nn.Module):
    def __init__(self, dataset, args, initialize=True):
        super(Model, self).__init__()
        self.encoder = Encoder(args, dataset)
        if args.n_layers > 0:
            self.decoder = Decoder(args)
        else:
            self.decoder = lambda x, *args: x  # identity function, which ignores all arguments except the first one

        if args.graph_mode == "sequential":
            self.head = SequentialHead(dataset, args, initialize)
        elif args.graph_mode == "node-centric":
            self.head = NodeCentricHead(dataset, args, initialize)
        elif args.graph_mode == "labeled-edge":
            self.head = LabeledEdgeHead(dataset, args, initialize)

        self.query_length = args.query_length
        self.dataset = dataset
        self.args = args

    def forward(self, batch, inference=False, **kwargs):
        every_input, word_lens = batch["every_input"]
        decoder_lens = self.query_length * word_lens
        batch_size, input_len = every_input.size(0), every_input.size(1)
        device = every_input.device

        encoder_mask = create_padding_mask(batch_size, input_len, word_lens, device)
        decoder_mask = create_padding_mask(batch_size, self.query_length * input_len, decoder_lens, device)

        encoder_output, decoder_input = self.encoder(batch["input"], batch["char_form_input"], batch["input_scatter"], input_len)

        decoder_output = self.decoder(decoder_input, encoder_output, decoder_mask, encoder_mask)

        if inference:
            return self.head.predict(encoder_output, decoder_output, encoder_mask, decoder_mask, batch)
        else:
            return self.head(encoder_output, decoder_output, encoder_mask, decoder_mask, batch)

    def get_params_for_optimizer(self, args):
        encoder_decay, encoder_no_decay = self.get_encoder_parameters(args.n_encoder_layers)
        decoder_decay, decoder_no_decay = self.get_decoder_parameters()

        parameters = [{"params": p, "weight_decay": args.encoder_weight_decay} for p in encoder_decay]
        parameters += [{"params": p, "weight_decay": 0.0} for p in encoder_no_decay]
        parameters += [
            {"params": decoder_decay, "weight_decay": args.decoder_weight_decay},
            {"params": decoder_no_decay, "weight_decay": 0.0},
        ]
        return parameters

    def get_decoder_parameters(self):
        no_decay = ["bias", "LayerNorm.weight", "_norm.weight"]
        decay_params = (p for name, p in self.named_parameters() if not any(nd in name for nd in no_decay) and not name.startswith("encoder.bert") and p.requires_grad)
        no_decay_params = (p for name, p in self.named_parameters() if any(nd in name for nd in no_decay) and not name.startswith("encoder.bert") and p.requires_grad)

        return decay_params, no_decay_params

    def get_encoder_parameters(self, n_layers):
        no_decay = ["bias", "LayerNorm.weight", "_norm.weight"]
        decay_params = [
            [p for name, p in self.named_parameters() if not any(nd in name for nd in no_decay) and name.startswith(f"encoder.bert.encoder.layer.{n_layers - 1 - i}.") and p.requires_grad] for i in range(n_layers)
        ]
        no_decay_params = [
            [p for name, p in self.named_parameters() if any(nd in name for nd in no_decay) and name.startswith(f"encoder.bert.encoder.layer.{n_layers - 1 - i}.") and p.requires_grad] for i in range(n_layers)
        ]

        return decay_params, no_decay_params
