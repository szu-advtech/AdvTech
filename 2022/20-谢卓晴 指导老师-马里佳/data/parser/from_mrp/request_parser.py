#!/usr/bin/env python3
# coding=utf-8

import utility.parser_utils as utils
from data.parser.from_mrp.abstract_parser import AbstractParser


class RequestParser(AbstractParser):
    def __init__(self, sentences, args, fields):
        self.data = {i: {"id": str(i), "sentence": sentence} for i, sentence in enumerate(sentences)}

        sentences = [example["sentence"] for example in self.data.values()]
    
        for example in self.data.values():
            example["input"] = example["sentence"].strip().split(' ')
            example["token anchors"], offset = [], 0
            for token in example["input"]:
                example["token anchors"].append([offset, offset + len(token)])
                offset += len(token) + 1

        utils.create_bert_tokens(self.data, args.encoder)

        super(RequestParser, self).__init__(fields, self.data)
