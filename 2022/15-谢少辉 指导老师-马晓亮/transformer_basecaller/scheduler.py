import math

import torch.optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

class LearningRateScheduler(object):
    r"""
    Provides interface of learning rate scheduler.

    Note:
        Do not use this class directly, use one of the sub classes.
    """
    def __init__(self, optimizer, lr):
        self.optimizer = optimizer
        self.lr = lr

    def step(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def set_lr(optimizer, lr):
        for g in optimizer.param_groups:
            g['lr'] = lr

    def get_lr(self):
        for g in self.optimizer.param_groups:
            return g['lr']


class TriStageLRScheduler(LearningRateScheduler):
    r"""
    Tri-Stage Learning Rate Scheduler. Implement the learning rate scheduler in "SpecAugment"

    Args:
        optimizer (Optimizer): Optimizer.
        lr (float): Peak learning rate.
        init_lr_scale (float): Initial learning rate scale.
        final_lr_scale (float): Final learning rate scale.
        warmup_steps (int): Warmup the learning rate linearly for the first N updates.
        hold_steps (int): Hold the learning rate for the N updates.
        decay_steps (int): Decay the learning rate linearly for the first N updates.
    """
    def __init__(
            self,
            optimizer: Optimizer,
            lr: float,
            init_lr_scale: float,
            final_lr_scale: float,
            warmup_steps: int,
            hold_steps: int,
            decay_steps: int,
    ):

        super(TriStageLRScheduler, self).__init__(optimizer, lr)

        self.peak_lr = lr
        self.init_lr  = init_lr_scale * lr
        self.final_lr = final_lr_scale * lr

        self.warmup_steps = warmup_steps
        self.hold_steps = hold_steps
        self.decay_steps = decay_steps

        self.warmup_rate = (self.peak_lr - self.init_lr) / self.warmup_steps if self.warmup_steps != 0 else 0
        if decay_steps == 0:
            self.decay_factor = None
        else:
            self.decay_factor = -math.log(final_lr_scale) / self.decay_steps

        self.lr = self.init_lr
        self.set_lr(self.optimizer, self.lr) # 优化器初始学习率必须设置
        self.update_steps = 1 # 优化器初始学习率设置后, update_steps应该为1

    def _decide_stage(self):
        if self.update_steps < self.warmup_steps:
            return 0, self.update_steps

        offset = self.warmup_steps

        if self.update_steps < offset + self.hold_steps:
            return 1, self.update_steps - offset

        offset += self.hold_steps

        if self.update_steps <= offset + self.decay_steps:
            # decay stage
            return 2, self.update_steps - offset

        offset += self.decay_steps

        return 3, self.update_steps - offset

    def step(self):
        stage, steps_in_stage = self._decide_stage()

        if stage == 0:
            self.lr = self.init_lr + self.warmup_rate * steps_in_stage
        elif stage == 1:
            self.lr = self.peak_lr
        elif stage == 2:
            self.lr = self.peak_lr * math.exp(-self.decay_factor * steps_in_stage)
        elif stage == 3:
            self.lr = self.final_lr
        else:
            raise ValueError("Undefined stage")

        self.set_lr(self.optimizer, self.lr)
        self.update_steps += 1

        return self.lr


def get_tri_stage_scheduler(
        optimizer,
        num_warmup_steps,
        num_hold_steps,
        num_decay_steps,
        lr_end=1e-7,
        power=1.0,
        last_epoch=-1
):
    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < (num_warmup_steps + num_hold_steps):
            return 1.0
        elif current_step < (num_warmup_steps + num_hold_steps + num_decay_steps):
            lr_range = lr_init - lr_end
            pct_remaining = 1 - (current_step - num_warmup_steps - num_hold_steps) / num_decay_steps
            decay = lr_range * pct_remaining**power + lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
        else:
            return lr_end / lr_init  # as LambdaLR multiplies by lr_init

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_four_stage_scheduler(
        optimizer,
        num_warmup_steps,
        num_hold_steps,
        num_decay_steps,
        second_hold_steps,
        second_decay_steps,
        first_lr_end=1e-5,
        second_lr_end=1e-7,
        power1=1.0,
        power2=1.0,
        last_epoch=-1
):
    lr_init = optimizer.defaults["lr"]

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        elif current_step < (num_warmup_steps + num_hold_steps):
            return 1.0
        elif current_step < (num_warmup_steps + num_hold_steps + num_decay_steps):
            lr_range = lr_init - first_lr_end
            pct_remaining = 1 - (current_step - num_warmup_steps - num_hold_steps) / num_decay_steps
            decay = lr_range * pct_remaining**power1 + first_lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
        elif current_step < (num_warmup_steps + num_hold_steps + num_decay_steps + second_hold_steps):
            return first_lr_end / lr_init  # as LambdaLR multiplies by lr_init
        elif current_step < (num_warmup_steps + num_hold_steps + num_decay_steps + second_hold_steps + second_decay_steps):
            lr_range = first_lr_end - second_lr_end
            pct_remaining = 1 - (current_step - num_warmup_steps - num_hold_steps - num_decay_steps - second_hold_steps) / second_decay_steps
            decay = lr_range * pct_remaining ** power2 + second_lr_end
            return decay / lr_init  # as LambdaLR multiplies by lr_init
        else:
            return second_lr_end / lr_init
    return LambdaLR(optimizer, lr_lambda, last_epoch)
