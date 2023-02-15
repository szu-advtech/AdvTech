#!/usr/bin/env python3
# coding=utf-8

from data.parser.from_mrp.abstract_parser import AbstractParser
import utility.parser_utils as utils


class EvaluationParser(AbstractParser):
    def __init__(self, args, fields):
        path = args.test_data
        self.data = utils.load_dataset(path)

        for sentence in self.data.values():
            sentence["token anchors"] = [[a["from"], a["to"]] for a in sentence["token anchors"]]

        utils.create_bert_tokens(self.data, args.encoder)

        super(EvaluationParser, self).__init__(fields, self.data)
