import torch
from torch import nn


class NeRFLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 一个极小的数
        self.lambda_opa = 1e-3

    def forward(self, results, target, **kwargs):
        d = {}
        # L2 损失
        d['rgb'] = (results['rgb'] - target['rgb']) ** 2
        o = results['opacity'] + 1e-10
        # encourage opacity to be either 0 or 1 to avoid floater 鼓励不透明度为0或1，以避免浮动
        d['opacity'] = self.lambda_opa * (-o * torch.log(o))
        return d


class L1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, inputs, targets, mask=None):
        if mask is None:
            loss = self.loss(inputs, targets)
        else:
            loss = self.loss(inputs * mask, targets * mask)
        return loss


class MSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets, mask=None):
        if mask is None:
            loss = self.loss(inputs, targets)
        else:
            loss = self.loss(inputs * mask, targets * mask)
        return loss


class SmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = lambda x: torch.mean(torch.abs(x))

    def forward(self, inputs):
        L1 = self.loss(inputs[:, :, :-1] - inputs[:, :, 1:])
        L2 = self.loss(inputs[:, :-1, :] - inputs[:, 1:, :])
        L3 = self.loss(inputs[:, :-1, :-1] - inputs[:, 1:, 1:])
        L4 = self.loss(inputs[:, 1:, :-1] - inputs[:, :-1, 1:])
        return (L1 + L2 + L3 + L4) / 4


class EdgePreservingSmoothnessLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = lambda x: torch.mean(torch.abs(x))
        self.bilateral_filter = lambda x: torch.exp(-torch.abs(x).sum(-1) / 0.1)

    def forward(self, inputs, weights):
        w1 = self.bilateral_filter(weights[:, :, :-1] - weights[:, :, 1:])
        w2 = self.bilateral_filter(weights[:, :-1, :] - weights[:, 1:, :])
        w3 = self.bilateral_filter(weights[:, :-1, :-1] - weights[:, 1:, 1:])
        w4 = self.bilateral_filter(weights[:, 1:, :-1] - weights[:, :-1, 1:])

        L1 = self.loss(w1 * (inputs[:, :, :-1] - inputs[:, :, 1:]))
        L2 = self.loss(w2 * (inputs[:, :-1, :] - inputs[:, 1:, :]))
        L3 = self.loss(w3 * (inputs[:, :-1, :-1] - inputs[:, 1:, 1:]))
        L4 = self.loss(w4 * (inputs[:, 1:, :-1] - inputs[:, :-1, 1:]))
        return (L1 + L2 + L3 + L4) / 4
