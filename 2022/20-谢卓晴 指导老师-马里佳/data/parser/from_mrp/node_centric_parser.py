#!/usr/bin/env python3
# coding=utf-8

from data.parser.from_mrp.abstract_parser import AbstractParser
import utility.parser_utils as utils


class NodeCentricParser(AbstractParser):
    def __init__(self, args, part: str, fields, filter_pred=None, **kwargs):
        assert part == "training" or part == "validation"
        path = args.training_data if part == "training" else args.validation_data

        self.data = utils.load_dataset(path)
        utils.anchor_ids_from_intervals(self.data)

        self.node_counter, self.edge_counter, self.no_edge_counter = 0, 0, 0
        anchor_count, n_node_token_pairs = 0, 0

        for sentence_id, sentence in list(self.data.items()):
            for node in sentence["nodes"]:
                if "label" not in node:
                    del self.data[sentence_id]
                    break

        for node, _ in utils.node_generator(self.data):
            self.node_counter += 1

        # print(f"Number of unlabeled nodes: {unlabeled_count}", flush=True)

        utils.create_bert_tokens(self.data, args.encoder)

        # create edge vectors
        for sentence in self.data.values():
            N = len(sentence["nodes"])

            edge_count = utils.create_edges(sentence)
            self.edge_counter += edge_count
            # self.no_edge_counter += len([n for n in sentence["nodes"] if n["label"] in ["Source", "Target"]]) * len([n for n in sentence["nodes"] if n["label"] not in ["Source", "Target"]]) - edge_count
            self.no_edge_counter += N * (N - 1) - edge_count

            sentence["anchor edges"] = [N, len(sentence["input"]), []]
            sentence["source anchor edges"] = [N, len(sentence["input"]), []]  # dummy
            sentence["target anchor edges"] = [N, len(sentence["input"]), []]  # dummy
            sentence["anchored labels"] = [len(sentence["input"]), []]
            for i, node in enumerate(sentence["nodes"]):
                anchored_labels = []
                #if len(node["anchors"]) == 0:
                #    print(f"Empty node in {sentence['id']}", flush=True)

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

        super(NodeCentricParser, self).__init__(fields, self.data, filter_pred)

    @staticmethod
    def node_similarity_key(node):
        return tuple([node["label"]] + node["anchors"])
