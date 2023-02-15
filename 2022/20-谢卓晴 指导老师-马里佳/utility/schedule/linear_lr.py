#!/usr/bin/env python3
# coding=utf-8

import math


class LinearLr:
    def __init__(self, param_group, learning_rate: float, total_steps: int, delay: bool, multiplier: int):
        self.total_steps = total_steps
        self.delay_steps = total_steps / 20 if delay else 0
        self.max_lr = learning_rate
        self.steps = 0
        self.param_group = param_group
        self.decay_multiplier = multiplier

    def __call__(self, _):
        self.steps += 1

        if self.steps < self.delay_steps:
            lr = 0.0
        elif self.steps < self.total_steps / 10:
            lr = self.max_lr * (self.steps - self.delay_steps) / (self.total_steps / 10 - self.delay_steps)
        else:
            max_lr = self.max_lr - self.max_lr / self.decay_multiplier
            min_lr = self.max_lr / self.decay_multiplier
            lr = max_lr * (math.cos(math.pi * (self.steps - self.total_steps / 10) / (self.total_steps * 9 / 10)) + 1) / 2 + min_lr
            #lr = self.max_lr * (self.total_steps - self.steps) / (self.total_steps * 9 / 10)

        # Safety first!
        if lr < 0.0:
            lr = 0.0

        self.param_group["lr"] = lr

    def lr(self) -> float:
        return self.param_group["lr"]
