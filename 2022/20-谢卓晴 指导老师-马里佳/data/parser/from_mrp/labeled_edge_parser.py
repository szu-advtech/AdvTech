#!/usr/bin/env python3
# coding=utf-8

from data.parser.from_mrp.abstract_parser import AbstractParser
import utility.parser_utils as utils


class LabeledEdgeParser(AbstractParser):
    def __init__(self, args, part: str, fields, filter_pred=None, **kwargs):
        assert part == "training" or part == "validation"
        path = args.training_data if part == "training" else args.validation_data

        self.data = utils.load_dataset(path)
        utils.anchor_ids_from_intervals(self.data)

        self.node_counter, self.edge_counter, self.no_edge_counter = 0, 0, 0
        anchor_count, n_node_token_pairs = 0, 0

        for sentence_id, sentence in list(self.data.items()):
            for edge in sentence["edges"]:
                if "label" not in edge:
                    del self.data[sentence_id]
                    break

        for node, sentence in utils.node_generator(self.data):
            node["label"] = "Node"

            self.node_counter += 1

        utils.create_bert_tokens(self.data, args.encoder)

        # create edge vectors
        for sentence in self.data.values():
            assert sentence["tops"] == [0], sentence
            N = len(sentence["nodes"])

            edge_count = utils.create_edges(sentence)
            self.edge_counter += edge_count
            self.no_edge_counter += N * (N - 1) - edge_count

            sentence["nodes"] = sentence["nodes"][1:]
            N = len(sentence["nodes"])

            sentence["anchor edges"] = [N, len(sentence["input"]), []]
            sentence["source anchor edges"] = [N, len(sentence["input"]), []]  # dummy
            sentence["target anchor edges"] = [N, len(sentence["input"]), []]  # dummy
            sentence["anchored labels"] = [len(sentence["input"]), []]
            for i, node in enumerate(sentence["nodes"]):
                anchored_labels = []

                for anchor in node["anchors"]:
                    sentence["anchor edges"][-1].append((i, anchor))
                    anchored_labels.append((anchor, node["label"]))

                sentence["anchored labels"][1].append(anchored_labels)

                anchor_count += len(node["anchors"])
                n_node_token_pairs += len(sentence["input"])

            sentence["id"] = [sentence["id"]]

        self.anchor_freq = anchor_count / n_node_token_pairs
        self.source_anchor_freq = self.target_anchor_freq = 0.5  # dummy
        self.input_count = sum(len(sentence["input"]) for sentence in self.data.values())

        super(LabeledEdgeParser, self).__init__(fields, self.data, filter_pred)

    @staticmethod
    def node_similarity_key(node):
        return tuple([node["label"]] + node["anchors"])
