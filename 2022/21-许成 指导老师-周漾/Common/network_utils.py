import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag