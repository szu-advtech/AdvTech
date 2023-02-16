import model_genotype
from model_operation import *
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


class Cell(nn.Module):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        # the output of preprocess of the previous 2 layers should have same channels
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)     # half the size of HW
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index, in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob=0):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]           # input 1
            h2 = states[self._indices[2 * i + 1]]       # input 2
            op1 = self._ops[2 * i]                      # operation 1
            op2 = self._ops[2 * i + 1]                  # operation 2
            h1 = op1(h1)                                # out 1
            h2 = op2(h2)                                # out 2
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2                                 # out = out1 + out2
            states += [s]                               # out => h_index
        return torch.cat([states[i] for i in self._concat], dim=1)


class NetworkCIFAR(nn.Module):

    def __init__(self, C, num_classes, layers, auxiliary, genotype, increment=4, droprate=0.0):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        self.droprate = droprate

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
        )
        # C_curr: the output Channel num after h1 and h2 pass through a 1x1 conv
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                reduction = True
            else:
                reduction = False

            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)

            reduction_prev = reduction
            self.cells += [cell]
            # cell.multiplier is the num of concat output, each one has C_curr channels
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            # if using an auxiliary classifier for training in the position of 2/3 depth
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

            C_curr += increment

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        # output -> GAP (Global Average Pooling) -> FC
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear( C_prev, num_classes)

    def forward(self, input):
        logits_aux = torch.tensor(0)
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.droprate)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)        # auxiliary result
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))     # Formal result
        return logits, logits_aux


class AuxiliaryHeadCIFAR(nn.Module):

    def __init__(self, C, num_classes):
        """ assume that input is 8x8 """
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.AvgPool2d(5, stride=3, padding=0, count_include_pad=False),  # 2x2
            nn.Conv2d(C, 128, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 768, 2, bias=False), # 1x1
        )
        self.bn = nn.BatchNorm2d(768)
        self.relu = nn.ReLU(inplace=True)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        if x.size(0) > 1:
            x = self.bn(x)
        x = self.relu(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


if __name__ == '__main__':
    geno = model_genotype.decode(model_genotype.test_genome)
    net = NetworkCIFAR(C=24, num_classes=10, layers=11, auxiliary=True, genotype=geno)
    inputs = torch.rand([1, 3, 32, 32])
    output1, output2 = net(inputs)
    print(f'output1: {output1}')
    print(f'output2: {output2}')

