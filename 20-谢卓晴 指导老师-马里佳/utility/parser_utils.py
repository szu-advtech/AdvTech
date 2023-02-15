#!/usr/bin/env python3
# coding=utf-8

import json
from itertools import chain
from transformers import AutoTokenizer

from utility.subtokenize import subtokenize

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_dataset(path):
    data = {}
    with open(path, encoding="utf8") as f:
        for sentence in f.readlines():
            sentence = json.loads(sentence)
            data[sentence["id"]] = sentence

            if "nodes" not in sentence:
                sentence["nodes"] = []

            if "edges" not in sentence:
                sentence["edges"] = []

    for sample in list(data.values()):
        sample["sentence"] = sample["input"]
        sample["input"] = sample["sentence"].split(' ')
        sample["token anchors"], offset = [], 0
        for token in sample["input"]:
            sample["token anchors"].append({"from": offset, "to": offset + len(token)})
            offset += len(token) + 1
    return data


def node_generator(data):
    for d in data.values():
        for n in d["nodes"]:
            yield n, d


def anchor_ids_from_intervals(data):
    for node, sentence in node_generator(data):
        if "anchors" not in node:
            node["anchors"] = []
        node["anchors"] = sorted(node["anchors"], key=lambda a: (a["from"], a["to"]))
        node["token references"] = set()

        for anchor in node["anchors"]:
            for i, token_anchor in enumerate(sentence["token anchors"]):
                if token_anchor["to"] <= anchor["from"]:
                    continue
                if token_anchor["from"] >= anchor["to"]:
                    break

                node["token references"].add(i)

        node["anchor intervals"] = node["anchors"]
        node["anchors"] = sorted(list(node["token references"]))
        del node["token references"]

    for sentence in data.values():
        sentence["token anchors"] = [[a["from"], a["to"]] for a in sentence["token anchors"]]


def create_bert_tokens(data, encoder: str):
    tokenizer = AutoTokenizer.from_pretrained(encoder, use_fast=True)

    for sentence in data.values():
        sentence["bert input"], sentence["to scatter"] = subtokenize(sentence["input"], tokenizer)


def create_edges(sentence, label_f=None):
    N = len(sentence["nodes"])

    sentence["edge presence"] = [N, N, []]
    sentence["edge labels"] = [N, N, []]

    for e in sentence["edges"]:
        source, target = e["source"], e["target"]
        label = e["label"] if "label" in e else "none"

        if label_f is not None:
            label = label_f(label)

        sentence["edge presence"][-1].append((source, target, 1))
        sentence["edge labels"][-1].append((source, target, label))

    edge_counter = len(sentence["edge presence"][-1])
    return edge_counter
