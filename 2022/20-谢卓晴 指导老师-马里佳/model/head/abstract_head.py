#!/usr/bin/env python3
# coding=utf-8

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.module.edge_classifier import EdgeClassifier
from model.module.anchor_classifier import AnchorClassifier
from utility.cross_entropy import cross_entropy, binary_cross_entropy
from utility.hungarian_matching import get_matching, reorder, match_anchor, match_label
from utility.utils import create_padding_mask


class AbstractHead(nn.Module):
    def __init__(self, dataset, args, config, initialize: bool):
        super(AbstractHead, self).__init__()

        self.edge_classifier = self.init_edge_classifier(dataset, args, config, initialize)
        self.label_classifier = self.init_label_classifier(dataset, args, config, initialize)
        self.anchor_classifier = self.init_anchor_classifier(dataset, args, config, initialize, mode="anchor")
        self.source_anchor_classifier = self.init_anchor_classifier(dataset, args, config, initialize, mode="source_anchor")
        self.target_anchor_classifier = self.init_anchor_classifier(dataset, args, config, initialize, mode="target_anchor")

        self.query_length = args.query_length
        self.focal = args.focal
        self.dataset = dataset

    def forward(self, encoder_output, decoder_output, encoder_mask, decoder_mask, batch):
        output = {}

        decoder_lens = self.query_length * batch["every_input"][1]
        output["label"] = self.forward_label(decoder_output)
        output["anchor"] = self.forward_anchor(decoder_output, encoder_output, encoder_mask, mode="anchor")  # shape: (B, T_l, T_w)
        output["source_anchor"] = self.forward_anchor(decoder_output, encoder_output, encoder_mask, mode="source_anchor")  # shape: (B, T_l, T_w)
        output["target_anchor"] = self.forward_anchor(decoder_output, encoder_output, encoder_mask, mode="target_anchor")  # shape: (B, T_l, T_w)

        cost_matrices = self.create_cost_matrices(output, batch, decoder_lens)
        matching = get_matching(cost_matrices)
        decoder_output = reorder(decoder_output, matching, batch["labels"][0].size(1))
        output["edge presence"], output["edge label"] = self.forward_edge(decoder_output)

        return self.loss(output, batch, matching, decoder_mask)

    def predict(self, encoder_output, decoder_output, encoder_mask, decoder_mask, batch, **kwargs):
        every_input, word_lens = batch["every_input"]
        decoder_lens = self.query_length * word_lens
        batch_size = every_input.size(0)

        label_pred = self.forward_label(decoder_output)
        anchor_pred = self.forward_anchor(decoder_output, encoder_output, encoder_mask, mode="anchor")  # shape: (B, T_l, T_w)
        source_anchor_pred = self.forward_anchor(decoder_output, encoder_output, encoder_mask, mode="source_anchor")  # shape: (B, T_l, T_w)
        target_anchor_pred = self.forward_anchor(decoder_output, encoder_output, encoder_mask, mode="target_anchor")  # shape: (B, T_l, T_w)

        labels = [[] for _ in range(batch_size)]
        anchors, source_anchors, target_anchors = [[] for _ in range(batch_size)], [[] for _ in range(batch_size)], [[] for _ in range(batch_size)]

        for b in range(batch_size):
            label_indices = self.inference_label(label_pred[b, :decoder_lens[b], :]).cpu()
            for t in range(label_indices.size(0)):
                label_index = label_indices[t].item()
                if label_index == 0:
                    continue

                decoder_output[b, len(labels[b]), :] = decoder_output[b, t, :]

                labels[b].append(label_index)
                if anchor_pred is None:
                    anchors[b].append(list(range(t // self.query_length, word_lens[b])))
                else:
                    anchors[b].append(self.inference_anchor(anchor_pred[b, t, :word_lens[b]]).cpu())

                if source_anchor_pred is None:
                    source_anchors[b].append(list(range(t // self.query_length, word_lens[b])))
                else:
                    source_anchors[b].append(self.inference_anchor(source_anchor_pred[b, t, :word_lens[b]]).cpu())

                if target_anchor_pred is None:
                    target_anchors[b].append(list(range(t // self.query_length, word_lens[b])))
                else:
                    target_anchors[b].append(self.inference_anchor(target_anchor_pred[b, t, :word_lens[b]]).cpu())

        decoder_output = decoder_output[:, : max(len(l) for l in labels), :]
        edge_presence, edge_labels = self.forward_edge(decoder_output)

        outputs = [
            self.parser.parse(
                {
                    "labels": labels[b],
                    "anchors": anchors[b],
                    "source anchors": source_anchors[b],
                    "target anchors": target_anchors[b],
                    "edge presence": self.inference_edge_presence(edge_presence, b),
                    "edge labels": self.inference_edge_label(edge_labels, b),
                    "id": batch["id"][b].cpu(),
                    "tokens": batch["every_input"][0][b, : word_lens[b]].cpu(),
                    "token intervals": batch["token_intervals"][b, :, :].cpu(),
                },
                **kwargs
            )
            for b in range(batch_size)
        ]

        return outputs

    def loss(self, output, batch, matching, decoder_mask):
        batch_size = batch["every_input"][0].size(0)
        device = batch["every_input"][0].device
        T_label = batch["labels"][0].size(1)
        T_input = batch["every_input"][0].size(1)
        T_edge = batch["edge_presence"].size(1)

        input_mask = create_padding_mask(batch_size, T_input, batch["every_input"][1], device)  # shape: (B, T_input)
        label_mask = create_padding_mask(batch_size, T_label, batch["labels"][1], device)  # shape: (B, T_label)
        edge_mask = torch.eye(T_label, T_label, device=device, dtype=torch.bool).unsqueeze(0)  # shape: (1, T_label, T_label)
        edge_mask = edge_mask | label_mask.unsqueeze(1) | label_mask.unsqueeze(2)  # shape: (B, T_label, T_label)
        if T_edge != T_label:
            edge_mask = F.pad(edge_mask, (T_edge - T_label, 0, T_edge - T_label, 0), value=0)
        edge_label_mask = (batch["edge_presence"] == 0) | edge_mask

        if output["edge label"] is not None:
            batch["edge_labels"] = (
                batch["edge_labels"][0][:, :, :, :output["edge label"].size(-1)],
                batch["edge_labels"][1],
            )

        losses = {}
        losses.update(self.loss_label(output, batch, decoder_mask, matching))
        losses.update(self.loss_anchor(output, batch, input_mask, matching, mode="anchor"))
        losses.update(self.loss_anchor(output, batch, input_mask, matching, mode="source_anchor"))
        losses.update(self.loss_anchor(output, batch, input_mask, matching, mode="target_anchor"))
        losses.update(self.loss_edge_presence(output, batch, edge_mask))
        losses.update(self.loss_edge_label(output, batch, edge_label_mask.unsqueeze(-1)))

        stats = {f"{key}": value.detach().cpu().item() for key, value in losses.items()}
        total_loss = sum(losses.values()) / len(losses)

        return total_loss, stats

    @torch.no_grad()
    def create_cost_matrices(self, output, batch, decoder_lens):
        batch_size = len(batch["labels"][1])
        decoder_lens = decoder_lens.cpu()

        matrices = []
        for b in range(batch_size):
            label_cost_matrix = self.label_cost_matrix(output, batch, decoder_lens, b)
            anchor_cost_matrix = self.anchor_cost_matrix(output, batch, decoder_lens, b)

            cost_matrix = label_cost_matrix * anchor_cost_matrix
            matrices.append(cost_matrix.cpu())

        return matrices

    def init_edge_classifier(self, dataset, args, config, initialize: bool):
        if not config["edge presence"] and not config["edge label"]:
            return None
        return EdgeClassifier(dataset, args, initialize, presence=config["edge presence"], label=config["edge label"])

    def init_label_classifier(self, dataset, args, config, initialize: bool):
        if not config["label"]:
            return None

        classifier = nn.Sequential(
            nn.Dropout(args.dropout_label),
            nn.Linear(args.hidden_size, len(dataset.label_field.vocab) + 1, bias=True)
        )
        if initialize:
            classifier[1].bias.data = dataset.label_freqs.log()

        return classifier

    def init_anchor_classifier(self, dataset, args, config, initialize: bool, mode="anchor"):
        if not config[mode]:
            return None

        return AnchorClassifier(dataset, args, initialize, mode=mode)

    def forward_edge(self, decoder_output):
        if self.edge_classifier is None:
            return None, None
        return self.edge_classifier(decoder_output)

    def forward_label(self, decoder_output):
        if self.label_classifier is None:
            return None
        return torch.log_softmax(self.label_classifier(decoder_output), dim=-1)

    def forward_anchor(self, decoder_output, encoder_output, encoder_mask, mode="anchor"):
        classifier = getattr(self, f"{mode}_classifier")
        if classifier is None:
            return None
        return classifier(decoder_output, encoder_output, encoder_mask)

    def inference_label(self, prediction):
        prediction = prediction.exp()
        return torch.where(
            prediction[:, 0] > prediction[:, 1:].sum(-1),
            torch.zeros(prediction.size(0), dtype=torch.long, device=prediction.device),
            prediction[:, 1:].argmax(dim=-1) + 1
        )

    def inference_anchor(self, prediction):
        return prediction.sigmoid()

    def inference_edge_presence(self, prediction, example_index: int):
        if prediction is None:
            return None

        N = prediction.size(1)
        mask = torch.eye(N, N, device=prediction.device, dtype=torch.bool)
        return prediction[example_index, :, :].sigmoid().masked_fill(mask, 0.0).cpu()

    def inference_edge_label(self, prediction, example_index: int):
        if prediction is None:
            return None
        return prediction[example_index, :, :, :].cpu()

    def loss_edge_presence(self, prediction, target, mask):
        if self.edge_classifier is None or prediction["edge presence"] is None:
            return {}
        return {"edge presence": binary_cross_entropy(prediction["edge presence"], target["edge_presence"].float(), mask)}

    def loss_edge_label(self, prediction, target, mask):
        if self.edge_classifier is None or prediction["edge label"] is None:
            return {}
        return {"edge label": binary_cross_entropy(prediction["edge label"], target["edge_labels"][0].float(), mask)}

    def loss_label(self, prediction, target, mask, matching):
        if self.label_classifier is None or prediction["label"] is None:
            return {}

        prediction = prediction["label"]
        target = match_label(
            target["labels"][0], matching, prediction.shape[:-1], prediction.device, self.query_length
        )
        return {"label": cross_entropy(prediction, target, mask, focal=self.focal)}

    def loss_anchor(self, prediction, target, mask, matching, mode="anchor"):
        if getattr(self, f"{mode}_classifier") is None or prediction[mode] is None:
            return {}

        prediction = prediction[mode]
        target, anchor_mask = match_anchor(target[mode], matching, prediction.shape, prediction.device)
        mask = anchor_mask.unsqueeze(-1) | mask.unsqueeze(-2)
        return {mode: binary_cross_entropy(prediction, target.float(), mask)}

    def label_cost_matrix(self, output, batch, decoder_lens, b: int):
        if output["label"] is None:
            return 1.0

        target_labels = batch["anchored_labels"][b]  # shape: (num_nodes, num_inputs, num_classes)
        label_prob = output["label"][b, : decoder_lens[b], :].exp().unsqueeze(0)  # shape: (1, num_queries, num_classes)
        tgt_label = target_labels.repeat_interleave(self.query_length, dim=1)  # shape: (num_nodes, num_queries, num_classes)
        cost_matrix = ((tgt_label * label_prob).sum(-1) * label_prob[:, :, 1:].sum(-1)).t().sqrt()  # shape: (num_queries, num_nodes)

        return cost_matrix

    def anchor_cost_matrix(self, output, batch, decoder_lens, b: int):
        if output["anchor"] is None:
            return 1.0

        num_nodes = batch["labels"][1][b]
        word_lens = batch["every_input"][1]
        target_anchors, _ = batch["anchor"]
        pred_anchors = output["anchor"].sigmoid()

        tgt_align = target_anchors[b, : num_nodes, : word_lens[b]]  # shape: (num_nodes, num_inputs)
        align_prob = pred_anchors[b, : decoder_lens[b], : word_lens[b]]  # shape: (num_queries, num_inputs)
        align_prob = align_prob.unsqueeze(1).expand(-1, num_nodes, -1)  # shape: (num_queries, num_nodes, num_inputs)
        align_prob = torch.where(tgt_align.unsqueeze(0).bool(), align_prob, 1.0 - align_prob)  # shape: (num_queries, num_nodes, num_inputs)
        cost_matrix = align_prob.log().mean(-1).exp()  # shape: (num_queries, num_nodes)
        return cost_matrix
