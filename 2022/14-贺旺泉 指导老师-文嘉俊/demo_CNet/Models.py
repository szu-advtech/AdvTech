import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import numpy as np

from Aggregation import *
from torch.nn.modules.utils import _pair
from affinity_loss_ori import *

class CrossEntropy2d(nn.Module):
    def __init__(self, size_average=True, ignore_label=255):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label

    def forward(self, predict, target, weight=None):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()        
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        loss = F.cross_entropy(predict, target, weight=weight, size_average=self.size_average)
        return loss

class myLoss(nn.Module):
    def __init__(self, num_classes, down_sample_size):
        super(myLoss, self).__init__()
        self.main_loss = CrossEntropy2d()
        self.affinity_loss = AffinityLoss(num_classes=num_classes, down_sample_size=down_sample_size)

    def forward(self, predict, context_prior_map, target, weight=None):
        loss = self.main_loss(predict, target) + self.affinity_loss(context_prior_map, target)
        return loss


def adjust_learning_rate(optimizer,base_lr, i_iter, max_iter, power=0.9):
    lr = base_lr * ((1 - float(i_iter) / max_iter) ** (power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10


def scaled_l2(X, C, S):
    """
    scaled_l2 distance
    Args:
        X (b*n*d):  original feature input
        C (k*d):    code words, with k codes, each with d dimension
        S (k):      scale cofficient
    Return:
        D (b*n*k):  relative distance to each code
    Note:
        apparently the X^2 + C^2 - 2XC computation is 2x faster than
        elementwise sum, perhaps due to friendly cache in gpu
    """
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert S.numel() == C.shape[0], "scale, codeword num mismatch"

    b, n, d = X.shape
    X = X.view(-1, d)  # [bn, d]
    Ct = C.t()  # [d, k]
    X2 = X.pow(2.0).sum(-1, keepdim=True)  # [bn, 1]
    C2 = Ct.pow(2.0).sum(0, keepdim=True)  # [1, k]
    norm = X2 + C2 - 2.0 * X.mm(Ct)  # [bn, k]
    scaled_norm = S * norm
    D = scaled_norm.view(b, n, -1)  # [b, n, k]
    return D


def aggregate(A, X, C):
    """
    aggregate residuals from N samples
    Args:
        A (b*n*k):  weight of each feature contribute to code residual
        X (b*n*d):  original feature input
        C (k*d):    code words, with k codes, each with d dimension
    Return:
        E (b*k*d):  residuals to each code
    """
    assert X.shape[-1] == C.shape[-1], "input, codeword feature dim mismatch"
    assert A.shape[:2] == X.shape[:2], "weight, input dim mismatch"
    X = X.unsqueeze(2)  # [b, n, d] -> [b, n, 1, d]
    C = C[None, None, ...]  # [k, d] -> [1, 1, k, d]
    A = A.unsqueeze(-1)  # [b, n, k] -> [b, n, k, 1]
    R = (X - C) * A  # [b, n, k, d]
    E = R.sum(dim=1)  # [b, k, d]
    return E


class DilatedFCN(nn.Module):
    def __init__(self, num_features=103, num_classes=9, conv_features=64):
        super(DilatedFCN, self).__init__()
        self.conv0 = nn.Conv2d(num_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=1,
                               bias=True)
        self.conv1 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=2,
                               bias=True)
        self.conv2 = nn.Conv2d(conv_features, conv_features, kernel_size=3, stride=1, padding=0, dilation=3,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv_cls = nn.Conv2d(conv_features, num_classes, kernel_size=1, stride=1, padding=0,
                                  bias=True)

    def forward(self, x):
        interpolation = nn.UpsamplingBilinear2d(size=x.shape[2:4])

        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.avgpool(x)
        x = self.relu(self.conv2(x))
        x = self.conv_cls(x)
        x = interpolation(x)
        return x


class CNet(nn.Module):
    """implementation with CNet."""
    def __init__(self,
                 num_features,
                 prior_size,
                 num_classes,
                 prior_channels = 64,
                 am_kerner_size=3,
                 group=1,
                 enable_auxiliary_loss=False,
                 drop_out_ratio = 0.1,
                 **kwargs):
        super(CNet, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.prior_channels = prior_channels
        self.int_channel = num_features

        self.prior_size = prior_size

        self.am_kerner_size = am_kerner_size

        # DilatedFCN
        self.conv0 = nn.Conv2d(self.int_channel, prior_channels, kernel_size=5, stride=1, padding=0, dilation=1,
                               bias=True)
        self.conv1 = nn.Conv2d(prior_channels, prior_channels, kernel_size=5, stride=1, padding=0, dilation=2,
                               bias=True)
        self.conv2 = nn.Conv2d(prior_channels, prior_channels, kernel_size=5, stride=1, padding=0, dilation=3,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        # Aggregation
        # self.aggregation = AggregationModule(prior_channels, prior_channels,
        #                                            am_kerner_size)
        self.aggregation = AggregationModule_Group(prior_channels, prior_channels,
                                                   am_kerner_size)

        self.prior_conv = nn.Sequential(nn.Conv2d(self.prior_channels, np.prod(self.prior_size),
                                   kernel_size=1,
                                   padding=0,
                                   stride=1,
                                   groups=group))

        self.BN_prior = nn.BatchNorm2d(np.prod(self.prior_size))
        #类内
        self.intra_conv = nn.Conv2d(self.prior_channels, self.prior_channels,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1)
        #类间
        self.inter_conv = nn.Conv2d(self.prior_channels, self.prior_channels,
                                    kernel_size=1,
                                    padding=0,
                                    stride=1)
        #Concat学习
        self.bottleneck = nn.Conv2d(self.prior_channels + self.prior_channels * 2,
                                    self.prior_channels,
                                    kernel_size=5,
                                    padding=1)
        #分类
        self.cls_seg = nn.Sequential(nn.Dropout(drop_out_ratio),
                                     nn.Conv2d(self.prior_channels, num_classes,kernel_size=1)
                                    )
        # 辅助分类
        # if enable_auxiliary_loss:
        #    self.auxlayer = AUXFHead(
        #
        #    )

        self.enable_anxiliaryloss = enable_auxiliary_loss

    def forward(self, inputs):

        # interpolation = nn.UpsamplingBilinear2d(size=inputs.shape[2:4])
        x = inputs
        batch_size, channels, height, width = x.size()
        x = self.relu(self.conv0(x))
        x = self.relu(self.conv1(x))
        x = self.avgpool(x)
        x = self.relu(self.conv2(x))
        x_half = self.avgpool(x)

        # assert self.prior_size[0] == height and self.prior_size[1] == width

        xt = self.aggregation(x_half)
        # generate prior map
        context_prior_map = self.BN_prior(self.prior_conv(xt))
        context_prior_map = context_prior_map.view(batch_size, np.prod(self.prior_size),-1)
        context_prior_map = context_prior_map.permute(0, 2, 1)
        context_prior_map = torch.sigmoid(context_prior_map)

        # reshape x from B×C1×N to B×N×C1
        xt = xt.view(batch_size, self.prior_channels, -1)
        xt = xt.permute(0, 2, 1)

        # 类内上下文
        intra_context = torch.bmm(context_prior_map, xt)
        intra_context = intra_context.div(np.prod(self.prior_size))
        intra_context = intra_context.permute(0, 2, 1).contiguous()
        intra_context = intra_context.view(batch_size, self.prior_channels,
                                           self.prior_size[0],
                                           self.prior_size[1])
        intra_context = self.intra_conv(intra_context)

        #类间上下文
        inter_context_prior_map = 1 - context_prior_map
        inter_context = torch.bmm(inter_context_prior_map, xt)
        inter_context = inter_context.div(np.prod(self.prior_size))
        inter_context = inter_context.permute(0, 2, 1).contiguous()
        inter_context = inter_context.view(batch_size, self.prior_channels,
                                           self.prior_size[0],
                                           self.prior_size[1])
        inter_context = self.inter_conv(inter_context)

        #Concat
        concat_x = torch.cat([x_half, intra_context, inter_context], dim=1)
        output = self.bottleneck(concat_x)
        output = self.cls_seg(output)

        logit_list = F.upsample(output,inputs.size()[2:],mode="bilinear", align_corners=False)

        return logit_list, context_prior_map

