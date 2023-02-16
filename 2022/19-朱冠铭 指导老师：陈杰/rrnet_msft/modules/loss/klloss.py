import torch.nn as nn
from .functional import kl_loss
import torch_npu


class KLLoss(nn.Module):
    def __init__(self, factor):
        super(KLLoss, self).__init__()
        self.factor = factor

    def forward(self, ori_feats, projected_feats, hms, whs, inds):
        return kl_loss(ori_feats, projected_feats, hms, whs, inds, self.factor)
