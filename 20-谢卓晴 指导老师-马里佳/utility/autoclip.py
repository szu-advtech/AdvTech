#!/usr/bin/env python3
# coding=utf-8

import torch
import torch.nn as nn


class AutoClip:
    def __init__(self, parameters, initial_clipping=0.1, percentile=50, history_len=1000):
        self.parameters = list(parameters)
        self.grad_history = [torch.full([history_len], initial_clipping) for _ in self.parameters]

        self.index = 0
        self.history_len = history_len
        self.percentile = percentile

    @torch.no_grad()
    def __call__(self):
        self._add_to_history(self.parameters)

        grad_norms = []
        for parameter, history in zip(self.parameters, self.grad_history):
            if parameter.grad is None or not parameter.grad.abs().sum().is_nonzero():
                continue

            clip_value = self._get_percentile(history, self.percentile)
            grad_norms.append(nn.utils.clip_grad_norm_(parameter, clip_value).item())

        return sum(grad_norms) / len(grad_norms)

    def _add_to_history(self, parameters):
        for i, param in enumerate(parameters):
            if param.grad is None or not param.grad.abs().sum().is_nonzero():
                continue

            self.grad_history[i][self.index] = param.grad.data.norm(2)

        self.index = (self.index + 1) % self.history_len

    def _get_percentile(self, tensor, percentile):
        k = 1 + round(0.01 * percentile * (tensor.numel() - 1))
        return tensor.kthvalue(k).values.item()
