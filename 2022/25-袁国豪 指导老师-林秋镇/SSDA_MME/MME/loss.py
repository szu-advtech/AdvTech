from typing import Any, Optional, Tuple

import torch
from torch.autograd import Function
import torch.nn.functional as F

"""
class Gradreverse(Function):
    def __init__(self,lambd):
        self.lambd=lambd

    @staticmethod
    def forward(self,x):
        return x.view_as(x)

    @staticmethod
    def backward(self,grad_output):
        return grad_output*-self.lambd

def grad_reverse(x,lambd=1.0):
    return Gradreverse.apply(x,lambd)
"""

class GradReverse(torch.autograd.Function):
    def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
        ctx.coeff = coeff
        output = input * 1.0
        return output

    def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        return grad_output.neg() * ctx.coeff, None


def grad_reverse(x, coeff):
    return GradReverse.apply(x, coeff)

def adentropy(F1,feat,lamda,eta=0.1):
    out_t1=F1(feat,reverse=True,eta=eta)
    out_t1=F.softmax(out_t1)
    loss_adent = lamda * torch.mean(torch.sum(out_t1 *(torch.log(out_t1 + 1e-5)), 1))
    return loss_adent
