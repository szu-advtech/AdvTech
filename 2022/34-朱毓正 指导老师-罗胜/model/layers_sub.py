import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PositionalEncoding_old(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.2, max_len=168):
        super(PositionalEncoding_old, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = nn.LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return self.dropout(x + self.norm(sublayer(x)))
        # return x + self.dropout(sublayer(self.norm(x)))
        # return self.dropout(sublayer(x))
        # return self.dropout(sublayer(x))


class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model, dropout):
        super(MultiHeadedAttention, self).__init__()
        self.d_model = d_model

        self.d_k = 32
        self.h = d_model // 32
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.dropout = nn.Dropout(dropout)
        self.att_weights = None

    def forward(self, q, k, v):

        bs = q.size(0)
        q, k, v = \
            [l(x).view(bs, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (q, k, v))]
        output, self.att_weights = self.attention(q, k, v)
        output = output.transpose(1, 2).contiguous() \
            .view(bs, -1, self.h * self.d_k)
        output_att = self.linears[-1](output)
        return output_att

    def attention(self, q, k, v):
        scores = torch.matmul(q, k.transpose(-2, -1)) \
                 / math.sqrt(self.d_k)

        scores = F.softmax(scores, dim=-1)

        return torch.matmul(scores, v), scores

    def continuous_penalty(self):
        win_len = self.att_weights.size(-1)
        # bs * heads * win_len * win_len
        scores = self.att_weights.view(-1, win_len, win_len)
        penalty = 0
        for i in range(scores.size(-1) - 1):
            t_s = torch.abs(scores[:, :, i] - scores[:, :, i + 1])
            penalty += (torch.sum(t_s) / (scores.size(0) * scores.size(1)))
        return penalty

    def continuous_sample_wise_penalty(self):
        win_len = self.att_weights.size(-1)
        # bs * heads * win_len * win_len
        scores = self.att_weights.view(-1, win_len, win_len)
        penalty = 0
        for i in range(scores.size(-2) - 1):
            t_s = torch.abs(scores[:, i, :] - scores[:, i + 1, :])
            penalty += (torch.sum(t_s) / (scores.size(0) * scores.size(1)))
        return penalty

    def get_plot_data(self):
        return self.att_weights


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
