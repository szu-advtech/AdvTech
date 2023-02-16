import torch
import torch.nn as nn

class AreaLoss(nn.Module):
    def __init__(self, topk=25):
        super(AreaLoss, self).__init__()

    def forward(self, attn):
        loss = torch.sum(attn) / (attn.shape[0] * attn.shape[1] * attn.shape[2])
        return loss