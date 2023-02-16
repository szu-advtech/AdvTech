#!/usr/bin/env python3
# coding=utf-8

from data.parser.to_mrp.abstract_parser import AbstractParser


class LabeledEdgeParser(AbstractParser):
    def __init__(self, *args):
        super().__init__(*args)
        self.source_id = self.dataset.edge_label_field.vocab.stoi["Source"]
        self.target_id = self.dataset.edge_label_field.vocab.stoi["Target"]

    def parse(self, prediction):
        output = {}

        output["id"] = self.dataset.id_field.vocab.itos[prediction["id"].item()]
        output["nodes"] = self.create_nodes(prediction)
        output["nodes"] = self.create_anchors(prediction, output["nodes"], join_contiguous=True, at_least_one=True)
        output["nodes"] = [{"id": 0}] + output["nodes"]
        output["edges"] = self.create_edges(prediction, output["nodes"])

        return output

    def create_nodes(self, prediction):
        return [{"id": i + 1} for i, l in enumerate(prediction["labels"])]

    def create_edges(self, prediction, nodes):
        N = len(nodes)
        edge_prediction = prediction["edge presence"][:N, :N]

        edges = []
        for target in range(1, N):
            if edge_prediction[0, target] >= 0.5:
                prediction["edge labels"][0, target, self.source_id] = float("-inf")
                prediction["edge labels"][0, target, self.target_id] = float("-inf")
                self.create_edge(0, target, prediction, edges, nodes)

        for source in range(1, N):
            for target in range(1, N):
                if source == target:
                    continue
                if edge_prediction[source, target] < 0.5:
                    continue
                for i in range(prediction["edge labels"].size(2)):
                    if i not in [self.source_id, self.target_id]:
                        prediction["edge labels"][source, target, i] = float("-inf")
                self.create_edge(source, target, prediction, edges, nodes)

        return edges

    def get_edge_label(self, prediction, source, target):
        return self.dataset.edge_label_field.vocab.itos[prediction["edge labels"][source, target].argmax(-1).item()]
