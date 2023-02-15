#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn.functional as F


class Batch:
    @staticmethod
    def build(data):
        fields = list(data[0].keys())
        transposed = {}
        for field in fields:
            if isinstance(data[0][field], tuple):
                transposed[field] = tuple(Batch._stack(field, [example[field][i] for example in data]) for i in range(len(data[0][field])))
            else:
                transposed[field] = Batch._stack(field, [example[field] for example in data])

        return transposed

    @staticmethod
    def _stack(field: str, examples):
        if field == "anchored_labels":
            return examples

        dim = examples[0].dim()

        if dim == 0:
            return torch.stack(examples)

        lengths = [max(example.size(i) for example in examples) for i in range(dim)]
        if any(length == 0 for length in lengths):
            return torch.LongTensor(len(examples), *lengths)

        examples = [F.pad(example, Batch._pad_size(example, lengths)) for example in examples]
        return torch.stack(examples)

    @staticmethod
    def _pad_size(example, total_size):
        return [p for i, l in enumerate(total_size[::-1]) for p in (0, l - example.size(-1 - i))]

    @staticmethod
    def index_select(batch, indices):
        filtered_batch = {}
        for key, examples in batch.items():
            if isinstance(examples, list) or isinstance(examples, tuple):
                filtered_batch[key] = [example.index_select(0, indices) for example in examples]
            else:
                filtered_batch[key] = examples.index_select(0, indices)

        return filtered_batch

    @staticmethod
    def to_str(batch):
        string = "\n".join([f"\t{name}: {Batch._short_str(item)}" for name, item in batch.items()])
        return string

    @staticmethod
    def to(batch, device):
        converted = {}
        for field in batch.keys():
            converted[field] = Batch._to(batch[field], device)
        return converted

    @staticmethod
    def _short_str(tensor):
        # unwrap variable to tensor
        if not torch.is_tensor(tensor):
            # (1) unpack variable
            if hasattr(tensor, "data"):
                tensor = getattr(tensor, "data")
            # (2) handle include_lengths
            elif isinstance(tensor, tuple) or isinstance(tensor, list):
                return str(tuple(Batch._short_str(t) for t in tensor))
            # (3) fallback to default str
            else:
                return str(tensor)

        # copied from torch _tensor_str
        size_str = "x".join(str(size) for size in tensor.size())
        device_str = "" if not tensor.is_cuda else " (GPU {})".format(tensor.get_device())
        strt = "[{} of size {}{}]".format(torch.typename(tensor), size_str, device_str)
        return strt

    @staticmethod
    def _to(tensor, device):
        if not torch.is_tensor(tensor):
            if isinstance(tensor, tuple):
                return tuple(Batch._to(t, device) for t in tensor)
            elif isinstance(tensor, list):
                return [Batch._to(t, device) for t in tensor]
            else:
                raise Exception(f"unsupported type of {tensor} to be casted to cuda")

        return tensor.to(device, non_blocking=True)
