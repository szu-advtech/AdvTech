#!/usr/bin/env python3
# coding=utf-8

import pickle

import torch

from data.parser.from_mrp.node_centric_parser import NodeCentricParser
from data.parser.from_mrp.labeled_edge_parser import LabeledEdgeParser
from data.parser.from_mrp.sequential_parser import SequentialParser
from data.parser.from_mrp.evaluation_parser import EvaluationParser
from data.parser.from_mrp.request_parser import RequestParser
from data.field.edge_field import EdgeField
from data.field.edge_label_field import EdgeLabelField
from data.field.field import Field
from data.field.mini_torchtext.field import Field as TorchTextField
from data.field.label_field import LabelField
from data.field.anchored_label_field import AnchoredLabelField
from data.field.nested_field import NestedField
from data.field.basic_field import BasicField
from data.field.bert_field import BertField
from data.field.anchor_field import AnchorField
from data.batch import Batch


def char_tokenize(word):
    return [c for i, c in enumerate(word)]  # if i < 10 or len(word) - i <= 10]


class Collate:
    def __call__(self, batch):
        batch.sort(key=lambda example: example["every_input"][0].size(0), reverse=True)
        return Batch.build(batch)


class Dataset:
    def __init__(self, args, verbose=True):
        self.verbose = verbose
        self.sos, self.eos, self.pad, self.unk = "<sos>", "<eos>", "<pad>", "<unk>"

        self.bert_input_field = BertField()
        self.scatter_field = BasicField()
        self.every_word_input_field = Field(lower=True, init_token=self.sos, eos_token=self.eos, batch_first=True, include_lengths=True)

        char_form_nesting = TorchTextField(tokenize=char_tokenize, init_token=self.sos, eos_token=self.eos, batch_first=True)
        self.char_form_field = NestedField(char_form_nesting, include_lengths=True)

        self.label_field = LabelField(preprocessing=lambda nodes: [n["label"] for n in nodes])
        self.anchored_label_field = AnchoredLabelField()

        self.id_field = Field(batch_first=True, tokenize=lambda x: [x])
        self.edge_presence_field = EdgeField()
        self.edge_label_field = EdgeLabelField()
        self.anchor_field = AnchorField()
        self.source_anchor_field = AnchorField()
        self.target_anchor_field = AnchorField()
        self.token_interval_field = BasicField()

        self.load_dataset(args)

    def log(self, text):
        if not self.verbose:
            return
        print(text, flush=True)

    def load_state_dict(self, args, d):
        for key, value in d["vocabs"].items():
            getattr(self, key).vocab = pickle.loads(value)

    def state_dict(self):
        return {
            "vocabs": {key: pickle.dumps(value.vocab) for key, value in self.__dict__.items() if hasattr(value, "vocab")}
        }

    def load_sentences(self, sentences, args):
        dataset = RequestParser(
            sentences, args,
            fields={
                "input": [("every_input", self.every_word_input_field), ("char_form_input", self.char_form_field)],
                "bert input": ("input", self.bert_input_field),
                "to scatter": ("input_scatter", self.scatter_field),
                "token anchors": ("token_intervals", self.token_interval_field),
                "id": ("id", self.id_field),
            },
        )

        self.every_word_input_field.build_vocab(dataset, min_freq=1, specials=[self.pad, self.unk, self.sos, self.eos])
        self.id_field.build_vocab(dataset, min_freq=1, specials=[])

        return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=Collate())

    def load_dataset(self, args):
        parser = {
            "sequential": SequentialParser,
            "node-centric": NodeCentricParser,
            "labeled-edge": LabeledEdgeParser
        }[args.graph_mode]

        train = parser(
            args, "training",
            fields={
                "input": [("every_input", self.every_word_input_field), ("char_form_input", self.char_form_field)],
                "bert input": ("input", self.bert_input_field),
                "to scatter": ("input_scatter", self.scatter_field),
                "nodes": ("labels", self.label_field),
                "anchored labels": ("anchored_labels", self.anchored_label_field),
                "edge presence": ("edge_presence", self.edge_presence_field),
                "edge labels": ("edge_labels", self.edge_label_field),
                "anchor edges": ("anchor", self.anchor_field),
                "source anchor edges": ("source_anchor", self.source_anchor_field),
                "target anchor edges": ("target_anchor", self.target_anchor_field),
                "token anchors": ("token_intervals", self.token_interval_field),
                "id": ("id", self.id_field),
            },
            filter_pred=lambda example: len(example.input) <= 256,
        )

        val = parser(
            args, "validation",
            fields={
                "input": [("every_input", self.every_word_input_field), ("char_form_input", self.char_form_field)],
                "bert input": ("input", self.bert_input_field),
                "to scatter": ("input_scatter", self.scatter_field),
                "nodes": ("labels", self.label_field),
                "anchored labels": ("anchored_labels", self.anchored_label_field),
                "edge presence": ("edge_presence", self.edge_presence_field),
                "edge labels": ("edge_labels", self.edge_label_field),
                "anchor edges": ("anchor", self.anchor_field),
                "source anchor edges": ("source_anchor", self.source_anchor_field),
                "target anchor edges": ("target_anchor", self.target_anchor_field),
                "token anchors": ("token_intervals", self.token_interval_field),
                "id": ("id", self.id_field),
            },
        )

        test = EvaluationParser(
            args,
            fields={
                "input": [("every_input", self.every_word_input_field), ("char_form_input", self.char_form_field)],
                "bert input": ("input", self.bert_input_field),
                "to scatter": ("input_scatter", self.scatter_field),
                "token anchors": ("token_intervals", self.token_interval_field),
                "id": ("id", self.id_field),
            },
        )

        del train.data, val.data, test.data  # TODO: why?
        for f in list(train.fields.values()) + list(val.fields.values()) + list(test.fields.values()):  # TODO: why?
            if hasattr(f, "preprocessing"):
                del f.preprocessing

        self.train_size = len(train)
        self.val_size = len(val)
        self.test_size = len(test)

        self.log(f"\n{self.train_size} sentences in the train split")
        self.log(f"{self.val_size} sentences in the validation split")
        self.log(f"{self.test_size} sentences in the test split")

        self.node_count = train.node_counter
        self.token_count = train.input_count
        self.edge_count = train.edge_counter
        self.no_edge_count = train.no_edge_counter
        self.anchor_freq = train.anchor_freq

        self.source_anchor_freq = train.source_anchor_freq if hasattr(train, "source_anchor_freq") else 0.5
        self.target_anchor_freq = train.target_anchor_freq if hasattr(train, "target_anchor_freq") else 0.5
        self.log(f"{self.node_count} nodes in the train split")

        self.every_word_input_field.build_vocab(val, test, min_freq=1, specials=[self.pad, self.unk, self.sos, self.eos])
        self.char_form_field.build_vocab(train, min_freq=1, specials=[self.pad, self.unk, self.sos, self.eos])
        self.char_form_field.nesting_field.vocab = self.char_form_field.vocab
        self.id_field.build_vocab(train, val, test, min_freq=1, specials=[])
        self.label_field.build_vocab(train)
        self.anchored_label_field.vocab = self.label_field.vocab
        self.edge_label_field.build_vocab(train)
        print(list(self.edge_label_field.vocab.freqs.keys()), flush=True)

        self.char_form_vocab_size = len(self.char_form_field.vocab)
        self.create_label_freqs(args)
        self.create_edge_freqs(args)

        self.log(f"Edge frequency: {self.edge_presence_freq*100:.2f} %")
        self.log(f"{len(self.label_field.vocab)} words in the label vocabulary")
        self.log(f"{len(self.anchored_label_field.vocab)} words in the anchored label vocabulary")
        self.log(f"{len(self.edge_label_field.vocab)} words in the edge label vocabulary")
        self.log(f"{len(self.char_form_field.vocab)} characters in the vocabulary")

        self.log(self.label_field.vocab.freqs)
        self.log(self.anchored_label_field.vocab.freqs)

        self.train = torch.utils.data.DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=Collate(),
            pin_memory=True,
            drop_last=True
        )
        self.train_size = len(self.train.dataset)

        self.val = torch.utils.data.DataLoader(
            val,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=Collate(),
            pin_memory=True,
        )
        self.val_size = len(self.val.dataset)

        self.test = torch.utils.data.DataLoader(
            test,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            collate_fn=Collate(),
            pin_memory=True,
        )
        self.test_size = len(self.test.dataset)

        if self.verbose:
            batch = next(iter(self.train))
            print(f"\nBatch content: {Batch.to_str(batch)}\n")
            print(flush=True)

    def create_label_freqs(self, args):
        n_rules = len(self.label_field.vocab)
        blank_count = (args.query_length * self.token_count - self.node_count)
        label_counts = [blank_count] + [
            self.label_field.vocab.freqs[self.label_field.vocab.itos[i]]
            for i in range(n_rules)
        ]
        label_counts = torch.FloatTensor(label_counts)
        self.label_freqs = label_counts / (self.node_count + blank_count)
        self.log(f"Label frequency: {self.label_freqs}")

    def create_edge_freqs(self, args):
        edge_counter = [
            self.edge_label_field.vocab.freqs[self.edge_label_field.vocab.itos[i]] for i in range(len(self.edge_label_field.vocab))
        ]
        edge_counter = torch.FloatTensor(edge_counter)
        self.edge_label_freqs = edge_counter / self.edge_count
        self.edge_presence_freq = self.edge_count / (self.edge_count + self.no_edge_count)
