#!/usr/bin/env python3
# coding=utf-8

from data.parser.from_mrp.abstract_parser import AbstractParser
import utility.parser_utils as utils


class SequentialParser(AbstractParser):
    def __init__(self, args, part: str, fields, filter_pred=None, **kwargs):
        assert part == "training" or part == "validation"
        path = args.training_data if part == "training" else args.validation_data

        self.data = utils.load_dataset(path)
        utils.anchor_ids_from_intervals(self.data)

        self.node_counter, self.edge_counter, self.no_edge_counter = 0, 0, 0
        anchor_count, source_anchor_count, target_anchor_count, n_node_token_pairs = 0, 0, 0, 0

        for sentence_id, sentence in list(self.data.items()):
            for node in sentence["nodes"]:
                if "label" not in node:
                    del self.data[sentence_id]
                    break

        for node, _ in utils.node_generator(self.data):
            node["target anchors"] = []
            node["source anchors"] = []

        for sentence in self.data.values():
            for e in sentence["edges"]:
                source, target = e["source"], e["target"]

                if sentence["nodes"][target]["label"] == "Target":
                    sentence["nodes"][source]["target anchors"] += sentence["nodes"][target]["anchors"]
                elif sentence["nodes"][target]["label"] == "Source":
                    sentence["nodes"][source]["source anchors"] += sentence["nodes"][target]["anchors"]

            for i, node in list(enumerate(sentence["nodes"]))[::-1]:
                if "label" not in node or node["label"] in ["Source", "Target"]:
                    del sentence["nodes"][i]
            sentence["edges"] = []

        for node, sentence in utils.node_generator(self.data):
            self.node_counter += 1

        utils.create_bert_tokens(self.data, args.encoder)

        # create edge vectors
        for sentence in self.data.values():
            N = len(sentence["nodes"])

            utils.create_edges(sentence)
            self.no_edge_counter += N * (N - 1)

            sentence["anchor edges"] = [N, len(sentence["input"]), []]
            sentence["source anchor edges"] = [N, len(sentence["input"]), []]
            sentence["target anchor edges"] = [N, len(sentence["input"]), []]

            sentence["anchored labels"] = [len(sentence["input"]), []]
            for i, node in enumerate(sentence["nodes"]):
                anchored_labels = []

                for anchor in node["anchors"]:
                    sentence["anchor edges"][-1].append((i, anchor))
                    anchored_labels.append((anchor, node["label"]))

                for anchor in node["source anchors"]:
                    sentence["source anchor edges"][-1].append((i, anchor))
                for anchor in node["target anchors"]:
                    sentence["target anchor edges"][-1].append((i, anchor))

                sentence["anchored labels"][1].append(anchored_labels)

                anchor_count += len(node["anchors"])
                source_anchor_count += len(node["source anchors"])
                target_anchor_count += len(node["target anchors"])
                n_node_token_pairs += len(sentence["input"])

            sentence["id"] = [sentence["id"]]

        self.anchor_freq = anchor_count / n_node_token_pairs
        self.source_anchor_freq = anchor_count / n_node_token_pairs
        self.target_anchor_freq = anchor_count / n_node_token_pairs
        self.input_count = sum(len(sentence["input"]) for sentence in self.data.values())

        super(SequentialParser, self).__init__(fields, self.data, filter_pred)

    @staticmethod
    def node_similarity_key(node):
        return tuple([node["label"]] + node["anchors"])
