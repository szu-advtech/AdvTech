import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.io as scio

import matplotlib.pyplot as plt
# from ..builder import LOSSES
from utils import weight_reduce_loss


# def cross_entropy(pred,
#                   label,
#                   weight=None,
#                   class_weight=None,
#                   reduction='mean',
#                   avg_factor=None,
#                   ignore_index=-100):
#     """The wrapper function for :func:`F.cross_entropy`"""
#     # class_weight is a manual rescaling weight given to each class.
#     # If given, has to be a Tensor of size C element-wise losses
#     loss = F.cross_entropy(
#         pred,
#         label,
#         weight=class_weight,
#         reduction='none',
#         ignore_index=ignore_index)
#
#     # apply weights and do the reduction
#     if weight is not None:
#         weight = weight.float()
#     loss = weight_reduce_loss(
#         loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
#
#     return loss


def _expand_onehot_labels(labels, label_weights, label_channels):
    """Expand onehot labels to match the size of prediction."""
    bin_labels = labels.new_full((labels.size(0), label_channels), 0)
    inds = torch.nonzero(labels >= 1, as_tuple=False).squeeze()
    if inds.numel() > 0:
        bin_labels[inds, labels[inds] - 1] = 1
    if label_weights is None:
        bin_label_weights = None
    else:
        bin_label_weights = label_weights.view(-1, 1).expand(
            label_weights.size(0), label_channels)
    return bin_labels, bin_label_weights

def binary_cross_entropy(pred,
                         label,
                         use_sigmoid=False,
                         weight=None,
                         reduction='mean',
                         avg_factor=None,
                         class_weight=None):
    """Calculate the binary CrossEntropy loss.

    Args:
        pred (torch.Tensor): The prediction with shape (N, 1).
        label (torch.Tensor): The learning label of the prediction.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        reduction (str, optional): The method used to reduce the loss.
            Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
        class_weight (list[float], optional): The weight for each class.

    Returns:
        torch.Tensor: The calculated loss
    """
    if pred.dim() != label.dim():
        label, weight = _expand_onehot_labels(label, weight, pred.size(-1))

    # weighted element-wise losses
    if weight is not None:
        weight = weight.float()
    if use_sigmoid:
        loss = F.binary_cross_entropy_with_logits(
            pred, label.float(), weight=class_weight, reduction='none')
    else:
        loss = F.binary_cross_entropy(
            pred, label.float(), weight=class_weight, reduction='none')
    # do the reduction for the weighted loss
    loss = weight_reduce_loss(
        loss, weight, reduction=reduction, avg_factor=avg_factor)

    return loss


# def _construct_ideal_affinity_matrix(self, label, label_size):
#     scaled_labels = F.interpolate(
#         label.float(), size=label_size, mode="nearest")
#     scaled_labels = scaled_labels.squeeze_().long()
#     scaled_labels[scaled_labels == 255] = self.num_classes
#     one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
#     one_hot_labels = one_hot_labels.view(
#         one_hot_labels.size(0), -1, self.num_classes + 1).float()
#     ideal_affinity_matrix = torch.bmm(one_hot_labels,
#                                       one_hot_labels.permute(0, 2, 1))
#     return ideal_affinity_matrix


# def losses(self, seg_logit, seg_label):
#     """Compute ``seg``, ``prior_map`` loss."""
#     seg_logit, context_prior_map = seg_logit
#     logit_size = seg_logit.shape[2:]
#     loss = dict()
#     loss.update(super(CPHead, self).losses(seg_logit, seg_label))
#     prior_loss = self.loss_prior_decode(
#         context_prior_map,
#         self._construct_ideal_affinity_matrix(seg_label, logit_size))
#     loss['loss_prior'] = prior_loss
#     return loss

class AffinityLoss(nn.Module):

    def __init__(self, num_classes, down_sample_size, reduction='mean', lambda_u=1.0, lambda_g=1.0,
                 align_corners=False):
        super(AffinityLoss, self).__init__()
        self.num_classes = num_classes
        self.down_sample_size = down_sample_size
        if isinstance(down_sample_size, int):
            self.down_sample_size = [down_sample_size] * 2
        self.reduction = reduction
        self.lambda_u = lambda_u
        self.lambda_g = lambda_g
        self.align_corners = align_corners

    def forward(self, context_prior_map, label):
        # unary loss
        A = self._construct_ideal_affinity_matrix(label, self.down_sample_size)
        unary_loss = binary_cross_entropy(context_prior_map, A)

        # global loss
        diagonal_matrix = (1 - torch.eye(A.shape[1])).to(A.get_device())
        vtarget = diagonal_matrix * A

        # true intra-class rate(recall)
        recall_part = torch.sum(context_prior_map * vtarget, dim=2)
        denominator = torch.sum(vtarget, dim=2)
        # denominator = torch.where(denominator <= 0, torch.ones_like(denominator), denominator)

        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        recall_part = recall_part.div_(denominator)
        recall_label = torch.ones_like(recall_part)
        recall_loss = binary_cross_entropy(recall_part, recall_label, reduction=self.reduction)

        # true inter-class rate(specificity)
        spec_part = torch.sum((1 - context_prior_map) * (1 - A), dim=2)
        denominator = torch.sum(1 - A, dim=2)
        # denominator = torch.where(denominator <= 0, torch.ones_like(denominator), denominator)

        denominator = denominator.masked_fill_(~(denominator > 0), 1)
        spec_part = spec_part.div_(denominator)
        spec_label = torch.ones_like(spec_part)
        spec_loss = binary_cross_entropy(spec_part, spec_label, reduction=self.reduction)

        # intra-class predictive value(precision)
        precision_part = torch.sum(context_prior_map * vtarget, dim=2)
        denominator = torch.sum(context_prior_map, dim=2)
        # denominator = torch.where(denominator <= 0, torch.ones_like(denominator), denominator)

        denominator = denominator.masked_fill_(~(denominator > 0), 1)

        precision_part = precision_part.div_(denominator)
        precision_label = torch.ones_like(precision_part)
        precision_loss = binary_cross_entropy(precision_part, precision_label, reduction=self.reduction)

        global_loss = recall_loss + spec_loss + precision_loss

        return self.lambda_u * unary_loss + self.lambda_g * global_loss

    def _construct_ideal_affinity_matrix(self, label, label_size):
        # down sample
        label = torch.unsqueeze(label, dim=1)
        # scaled_labels = label
        scaled_labels = F.interpolate(label.float(), size=label_size, mode="nearest")
        scaled_labels = torch.squeeze(scaled_labels,dim=1).long()
        # scaled_labels = scaled_labels.squeeze_().long()

        # ### plot context prior map
        # scaled_labels += 1
        # scaled_labels[scaled_labels == 256] = 0
        # one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
        # one_hot_labels[one_hot_labels[:,:,:,0] == 0 ]
        # one_hot_labels = one_hot_labels.view(
        #     one_hot_labels.size(0), -1, self.num_classes + 1).float()
        # # ideal affinity map
        # ideal_affinity_matrix = torch.bmm(one_hot_labels,
        #                                   one_hot_labels.permute(0, 2, 1))
        # return ideal_affinity_matrix

        scaled_labels[scaled_labels == 255] = self.num_classes
        # scaled_labels[scaled_labels == 255] = 0
        # to one-hot
        one_hot_labels = F.one_hot(scaled_labels, self.num_classes + 1)
        one_hot_labels = one_hot_labels.view(
            one_hot_labels.size(0), -1, self.num_classes + 1).float()
        # ideal affinity map
        ideal_affinity_matrix = torch.bmm(one_hot_labels,
                                          one_hot_labels.permute(0, 2, 1))
        ###### plot context_label
        # plot_context_prior_map = np.squeeze(ideal_affinity_matrix.cpu().data.numpy())
        # scio.savemat('ontext_prior_map.mat', {'ontext_prior_map': plot_context_prior_map})
        # plt.imshow(plot_context_prior_map)
        # plt.imsave('./Exp_PaviaU/Affinity_label_7.png',plot_context_prior_map)

        return ideal_affinity_matrix

