import torch
from data.field.mini_torchtext.field import RawField
from data.field.mini_torchtext.vocab import Vocab
from collections import Counter


class LabelField(RawField):
    def __self__(self, preprocessing):
        super(LabelField, self).__init__(preprocessing=preprocessing)
        self.vocab = None

    def build_vocab(self, *args, **kwargs):
        sources = []
        for arg in args:
            if isinstance(arg, torch.utils.data.Dataset):
                sources += [arg.get_examples(name) for name, field in arg.fields.items() if field is self]
            else:
                sources.append(arg)

        counter = Counter()
        for data in sources:
            for x in data:
                counter.update(x)

        self.vocab = Vocab(counter, specials=[])

    def process(self, example, device=None):
        tensor, lengths = self.numericalize(example, device=device)
        return tensor, lengths

    def numericalize(self, example, device=None):
        example = [self.vocab.stoi[x] + 1 for x in example]
        length = torch.LongTensor([len(example)], device=device).squeeze(0)
        tensor = torch.LongTensor(example, device=device)

        return tensor, length
