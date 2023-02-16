#!/usr/bin/env python3
# coding=utf-8

import torch
from data.parser.json_parser import example_from_json


class AbstractParser(torch.utils.data.Dataset):
    def __init__(self, fields, data, filter_pred=None):
        super(AbstractParser, self).__init__()

        self.examples = [example_from_json(d, fields) for _, d in sorted(data.items())]

        if isinstance(fields, dict):
            fields, field_dict = [], fields
            for field in field_dict.values():
                if isinstance(field, list):
                    fields.extend(field)
                else:
                    fields.append(field)

        if filter_pred is not None:
            make_list = isinstance(self.examples, list)
            self.examples = filter(filter_pred, self.examples)
            if make_list:
                self.examples = list(self.examples)

        self.fields = dict(fields)

        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]

    def __getitem__(self, i):
        item = self.examples[i]
        processed_item = {}
        for (name, field) in self.fields.items():
            if field is not None:
                processed_item[name] = field.process(getattr(item, name), device=None)
        return processed_item

    def __len__(self):
        return len(self.examples)

    def get_examples(self, attr):
        if attr in self.fields:
            for x in self.examples:
                yield getattr(x, attr)
