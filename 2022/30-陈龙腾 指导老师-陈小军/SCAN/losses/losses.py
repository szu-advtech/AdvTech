"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn import Parameter

EPS = 1e-8
import math
import numpy as np

def LabelSmoothing(true_labels, classes, smoothing=0.0):

    label_shape = torch.Size((true_labels.size(0), classes))
    # with torch.no_grad():
    true_dist = torch.empty(size=label_shape, device=true_labels.device)
    true_dist.fill_(smoothing / (classes - 1))
    true_dist=torch.add(true_dist,true_labels)
    _, index = torch.max(true_labels, 1)
    max_label=true_labels[[0],[1]].numpy()-smoothing
    true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), max_label[0])
    return true_dist


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, input, target, mask, weight, reduction='mean'):
        if not (mask != 0).any():
            raise ValueError('Mask in MaskedCrossEntropyLoss is all zeros.')
        target = torch.masked_select(target, mask)
        b, c = input.size()
        n = target.size(0)
        input = torch.masked_select(input, mask.view(b, 1)).view(n, c)
        return F.cross_entropy(input, target, weight=weight, reduction=reduction)


# class ConfidenceBasedCE(nn.Module):
#     def __init__(self, ct1, ct2, eta, topk, apply_class_balancing):
#         super(ConfidenceBasedCE, self).__init__()
#         self.loss = MaskedCrossEntropyLoss()
#         self.softmax = nn.Softmax(dim=1)
#         self.logsoftmax_func = nn.LogSoftmax(dim=1)
#         self.ct1 = ct1
#         self.ct2 = ct2
#         self.eta = eta
#         self.apply_class_balancing = apply_class_balancing
#         self.tau = None
#
#         # attention parameters
#         self.att = Parameter(torch.Tensor(10, 1)).cuda()
#         self.proj_param = nn.Parameter(torch.Tensor(10, 10)).cuda()
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         """Reinitialize learnable parameters."""
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_normal_(self.att, gain=gain)
#         nn.init.xavier_normal_(self.proj_param, gain=gain)
#
#     def forward(self, anchors_weak, anchors_strong, neighbors, labels):
#         """
#         Loss function during self-labeling
#
#         input: logits for original samples and for its strong augmentations
#         output: cross entropy
#         """
#
#         # get prob
#         weak_anchors_prob = self.softmax(anchors_weak)
#         neighbors_prob = neighbors.softmax(2)
#
#         # set ct1
#         max_prob_0, target_0 = torch.max(weak_anchors_prob, dim=1)
#         mask0 = max_prob_0 > self.ct1
#
#         weak_anchors_prob = weak_anchors_prob[mask0]
#         neighbors_prob = neighbors_prob[mask0]
#         anchors_strong = anchors_strong[mask0]
#         b, c = weak_anchors_prob.size()
#
#
#         att_score = torch.exp(-torch.norm(weak_anchors_prob.unsqueeze(1)-neighbors_prob, dim=2)**2)
#         att_val = att_score/att_score.sum(1).view(-1,1)
#         att_val = att_val.unsqueeze(2)
#         q_hat = torch.sum(att_val*neighbors_prob, dim=1)
#
#         # compute the tau
#         beta = torch.norm(weak_anchors_prob - q_hat, dim=1) ** 2
#         topk = max(int(self.eta * b), 1)
#         topk_min, _ = torch.topk(beta, topk, largest=False)
#         self.tau = topk_min[-1] / torch.exp(torch.tensor([-1.0]).cuda())
#         alpha = -torch.log(beta / self.tau)
#
#         # hardening based on alpha and mask0
#         q = []
#         for i in range(len(alpha)):
#             if alpha[i] > 1:
#                 qi = weak_anchors_prob[i] ** alpha[i]
#                 qi = qi / qi.sum(0)
#             else:
#                 qi = weak_anchors_prob[i]
#             q.append(qi.unsqueeze(0))
#         q = torch.cat(q, dim=0)
#
#         # set ct2, retrieve target and mask based on soft-label q
#         max_prob, target = torch.max(q, dim=1)
#         mask = max_prob > self.ct2
#         target_masked = torch.masked_select(target, mask.squeeze())
#         n = target_masked.size(0)
#
#         # Class balancing weights
#         if self.apply_class_balancing:
#             idx, counts = torch.unique(target_masked, return_counts=True)
#             weight = torch.ones(c).cuda()
#             freq = counts.float() / n
#             h = 1.02
#             weight[idx] = 1 / torch.log(h + freq)
#         else:
#             weight = None
#
#         # Loss
#         input = torch.masked_select(anchors_strong, mask.view(b, 1)).view(n, c)
#         input_prob = self.logsoftmax_func(input)
#         q_mask = torch.masked_select(q, mask.view(b, 1)).view(n, c)
#
#         if self.apply_class_balancing:
#              w_avg = weight.view(1, -1) / torch.sum(weight) * torch.mean(weight)
#              loss = -torch.sum(torch.sum(w_avg * q_mask * input_prob, dim=1), dim=0) / n  # add weight
#         else:
#              loss = -torch.sum(torch.sum( q_mask * input_prob, dim=1), dim=0) / n
#
#         # Loss
#         # loss = self.loss(anchors_strong, target, mask, weight=weight, reduction='mean')
#
#
#         return loss, target_masked, labels[mask0][mask], n

# ------------------------------
#          strategy 2
# ------------------------------

class ConfidenceBasedCE(nn.Module):
    def __init__(self, ct1, ct2, eta, topk, att_type, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax_func = nn.LogSoftmax(dim=1)
        self.ct1 = ct1
        self.ct2 = ct2
        self.eta = eta
        self.apply_class_balancing = apply_class_balancing
        self.tau = None

        # attention parameters
        if att_type == 'concat':
            self.att = Parameter(torch.Tensor(10*2, 1)).cuda()
        elif att_type == 'cross_product':
            self.att = Parameter(torch.Tensor(10, 1)).cuda()

        self.proj_param = nn.Parameter(torch.Tensor(10, 10)).cuda()
        self.reset_parameters()

        self.att_type = att_type

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.att, gain=gain)
        nn.init.xavier_normal_(self.proj_param, gain=gain)

    def forward(self, anchors_weak, anchors_strong, anchors, neighbors, labels, index):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """

        # get prob
        weak_anchors_prob = self.softmax(anchors_weak)
        neighbors_prob = neighbors.softmax(2)
        anchors_prob = self.softmax(anchors)

        # set ct1
        max_prob_0, target_0 = torch.max(weak_anchors_prob, dim=1)
        mask0 = max_prob_0 > self.ct1

        weak_anchors_prob = weak_anchors_prob[mask0]
        neighbors_prob = neighbors_prob[mask0]
        anchors_prob = anchors_prob[mask0]

        anchors_strong = anchors_strong[mask0]
        images_features = anchors[mask0]
        neighbors_features = neighbors[mask0]

        b, c = weak_anchors_prob.size()

        # attention based on probability
        # qiqj = torch.cat([weak_anchors_prob.unsqueeze(1).expand_as(neighbors_prob), neighbors_prob], dim=2)  # concat
        # qiqj = torch.einsum('ijk,ijk->ijk', weak_anchors_prob.unsqueeze(1).expand_as(neighbors_prob),
        #                     neighbors_prob)  # cross product
        # att_score = F.leaky_relu(torch.einsum('k,ijk->ij', self.att.squeeze(), qiqj)).squeeze()
        # att_val = att_score.softmax(dim=1)
        # q_hat = torch.sum(1.0 / 21 * neighbors_prob, dim=1)

        # attention based on feature
        zi  = torch.einsum('kk, ijk->ijk', self.proj_param, images_features.unsqueeze(1).expand_as(neighbors_features))
        zj = torch.einsum('kk, ijk->ijk', self.proj_param, neighbors_features)
        # zi = images_features.unsqueeze(1).expand_as(neighbors_features)
        # zj = neighbors_features
        if self.att_type=='concat':
            zizj = torch.cat([zi, zj], dim=2)  # concat
        elif self.att_type=='cross_product':
            zizj = torch.einsum('ijk,ijk->ijk', zi, zj)  # cross product

        att_score = F.leaky_relu(torch.einsum('k,ijk->ij', self.att.squeeze(), zizj))
        # att_score = att_score / 5  # temperature
        att_val = att_score.softmax(dim=1).unsqueeze(-1)
        q_hat = torch.sum(att_val * neighbors_prob, dim=1)


        # compute the tau
        beta = torch.norm(anchors_prob - q_hat, dim=1) ** 2
        topk = max(int(self.eta * b), 1)
        topk_min, _ = torch.topk(beta, topk, largest=False)
        self.tau = topk_min[-1] / torch.exp(torch.tensor([-1.0]).cuda())
        alpha = -torch.log(beta / self.tau)

        # hardening based on alpha and mask0
        q = []
        for i in range(len(alpha)):
            if alpha[i] > 1:
                qi = weak_anchors_prob[i] ** alpha[i]
                qi = qi / qi.sum(0)
            else:
                qi = weak_anchors_prob[i]
            q.append(qi.unsqueeze(0))
        q = torch.cat(q, dim=0)

        # set ct2, retrieve target and mask based on soft-label q
        max_prob, target = torch.max(q, dim=1)
        mask = max_prob > self.ct2
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts=True)
            weight = torch.ones(c).cuda()
            freq = counts.float() / n
            h = 1.02
            weight[idx] = 1 / torch.log(h + freq)
        else:
            weight = None

        # Loss
        input = torch.masked_select(anchors_strong, mask.view(b, 1)).view(n, c)
        input_prob = self.logsoftmax_func(input)
        q_mask = torch.masked_select(q, mask.view(b, 1)).view(n, c)

        if self.apply_class_balancing:
            w_avg = weight.view(1, -1) / torch.sum(weight) * torch.mean(weight)
            loss = -torch.sum(torch.sum(w_avg * q_mask * input_prob, dim=1), dim=0) / n  # add weight
        else:
            loss = -torch.sum(torch.sum(q_mask * input_prob, dim=1), dim=0) / n

        # Loss
        # loss = self.loss(anchors_strong, target, mask, weight=weight, reduction='mean')

        return loss, target_masked, labels[mask0][mask], n


# ------------------------------
#          strategy 3
# ------------------------------

# class ConfidenceBasedCE(nn.Module):
#     def __init__(self, ct1, ct2, eta, topk, att_type, apply_class_balancing):
#         super(ConfidenceBasedCE, self).__init__()
#         self.loss = MaskedCrossEntropyLoss()
#         self.softmax = nn.Softmax(dim=1)
#         self.logsoftmax_func = nn.LogSoftmax(dim=1)
#         self.ct1 = ct1
#         self.ct2 = ct2
#         self.eta = eta
#         self.apply_class_balancing = apply_class_balancing
#         self.tau = None
#         self.topk = topk
#
#         # attention parameters
#         if att_type == 'concat':
#             self.att = Parameter(torch.Tensor(8000, topk+1)).cuda()
#         elif att_type == 'cross_product':
#             self.att = Parameter(torch.Tensor(512, 1)).cuda()
#
#         self.proj_param = nn.Parameter(torch.Tensor(512, 512)).cuda()
#         self.reset_parameters()
#
#         self.att_type = att_type
#
#     def reset_parameters(self):
#         """Reinitialize learnable parameters."""
#         gain = nn.init.calculate_gain('relu')
#         nn.init.xavier_normal_(self.att, gain=gain)
#         nn.init.xavier_normal_(self.proj_param, gain=gain)
#
#     def forward(self, anchors_weak, anchors_strong, neighbors, images_features, neighbors_features, labels, index):
#         """
#         Loss function during self-labeling
#
#         input: logits for original samples and for its strong augmentations
#         output: cross entropy
#         """
#
#         # get prob
#         weak_anchors_prob = self.softmax(anchors_weak)
#         neighbors_prob = neighbors.softmax(2)
#
#         # set ct1
#         max_prob_0, target_0 = torch.max(weak_anchors_prob, dim=1)
#         mask0 = max_prob_0 > self.ct1
#
#         weak_anchors_prob = weak_anchors_prob[mask0]
#         neighbors_prob = neighbors_prob[mask0]
#         anchors_strong = anchors_strong[mask0]
#         images_features = images_features[mask0]
#         neighbors_features = neighbors_features[mask0]
#         b, c = weak_anchors_prob.size()
#
#         q_hat = []
#         for i in range(neighbors_prob.shape[0]):
#             ni = neighbors_prob[i]
#             max_np, _ = torch.max(ni, dim=1)
#             mask = max_np > 0.98
#             if torch.sum(mask) > 0.8:
#                 q_hat_ = torch.sum(torch.softmax(self.att[index][mask0][i][mask].unsqueeze(-1), dim=0)*ni[mask], dim=0).unsqueeze(0)
#             else:
#                 q_hat_ = ni[0].unsqueeze(0)
#             q_hat.append(q_hat_)
#
#         q_hat = torch.cat(q_hat, dim=0)
#
#         # the high confident with sharping and the low confident replace with neighbors
#         max_prob, target = torch.max(weak_anchors_prob, dim=1)
#         # mask_pos = max_prob > 0.99
#         # q_sr = weak_anchors_prob[mask_pos]**1.05
#         # q_sr = q_sr / q_sr.sum(dim=1).view(-1,1).expand_as(q_sr)
#         # weak_anchors_prob[mask_pos] = q_sr
#         mask = max_prob <= 0.99
#         weak_anchors_prob[mask] = q_hat[mask]
#
#         # set ct2, retrieve target and mask based on soft-label q
#         max_prob, target = torch.max(weak_anchors_prob, dim=1)
#         mask = max_prob > self.ct2
#         target_masked = torch.masked_select(target, mask.squeeze())
#         n = target_masked.size(0)
#
#         # Class balancing weights
#         if self.apply_class_balancing:
#             idx, counts = torch.unique(target_masked, return_counts=True)
#             weight = torch.ones(c).cuda()
#             freq = counts.float() / n
#             h = 1.02
#             weight[idx] = 1 / torch.log(h + freq)
#         else:
#             weight = None
#
#         # Loss
#         input = torch.masked_select(anchors_strong, mask.view(b, 1)).view(n, c)
#         input_prob = self.logsoftmax_func(input)
#         q_mask = weak_anchors_prob[mask]
#
#         if self.apply_class_balancing:
#             w_avg = weight.view(1, -1) / torch.sum(weight) * torch.mean(weight)
#             loss = -torch.sum(torch.sum(w_avg * q_mask * input_prob, dim=1), dim=0) / n  # add weight
#         else:
#             loss = -torch.sum(torch.sum(q_mask * input_prob, dim=1), dim=0) / n
#
#         # Loss
#         # loss = self.loss(anchors_strong, target, mask, weight=weight, reduction='mean')
#
#         return loss, target_masked, labels[mask0][mask], n

# import seaborn as sns
# sns.kdeplot(data=ratio.cpu().numpy())
# from matplotlib import pyplot as plt
# plt.show()

# print("n:", n)
# import torch.nn.functional as F
# q = F.one_hot(target, c).cuda()
# the method of scan's selflabel method
# max_prob_0, target_0 = torch.max(weak_anchors_prob, dim=1)
# n0 = torch.sum(max_prob_0 > self.threshold)


# loss_nw = F.cross_entropy(input, target_masked, reduction='mean')
# loss_w = self.loss(input_, target, mask, weight = weight, reduction='mean')
# loss_nw_my = -torch.sum(torch.sum(target_masked_onehot*self.logsoftmax_func(input), dim=1), dim=0)/n
# loss_w_my = -torch.sum(torch.sum(w_avg*target_masked_onehot*self.logsoftmax_func(input), dim=1), dim=0)/n

# # if epoch > 10:
# #     self.threshold = max(self.threshold-(epoch-10)/5000, 0.95)
# topk = 10
# neighbors_prob = neighbors.softmax(2)
# max_prob, _ = torch.max(neighbors_prob, dim = 2)
# _, ind = torch.sort(max_prob, dim=1, descending=True)
# n_prob = [neighbors_prob[i][ind[i]][:topk].unsqueeze(0) for i in range(len(ind))]
# neighbors_prob = torch.cat(n_prob, dim=0)
#
# weak_anchors_prob = self.softmax(anchors_weak)
# b, c = weak_anchors_prob.size()
# p_distances = torch.sum( torch.sum((weak_anchors_prob.unsqueeze(1)-neighbors_prob)**2, dim=2)/c, dim=1 )/topk
# _, ind = torch.sort(p_distances, dim=0, descending=False)
#
# ind = ind[:int(0.4*b)]
# mask = torch.zeros_like(p_distances, dtype=torch.bool)
# mask[ind] = True


class ConfidenceBasedCE_scan(nn.Module):
    def __init__(self, ct1, ct2, apply_class_balancing):
        super(ConfidenceBasedCE_scan, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax_func = nn.LogSoftmax(dim=1)
        self.threshold = ct2
        self.ct1 = ct1
        self.ct2 = ct2
        self.apply_class_balancing = apply_class_balancing
        self.tau = None

    def forward(self, anchors_weak, anchors_strong, neighbors, labels):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        b, c = weak_anchors_prob.size()
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts=True)
            freq = 1 / (counts.float() / n)
            weight = torch.ones(c).cuda()
            weight[idx] = freq

        else:
            weight = None

        # Loss
        loss = self.loss(input_, target, mask, weight=weight, reduction='mean')

        return loss, target_masked, labels[mask], n


def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ = torch.clamp(x, min=EPS)
        b = x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)

    if len(b.size()) == 2:  # Sample-wise entropy
        return -b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' % (len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, wo1, wo2, wo3, t, u, num_classes, entropy_weight=5.0, weight_t=1.0, alpha=0.1):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = None
        self.entropy_weight = entropy_weight
        self.class_num = num_classes
        self.tau_c = t

        self.mask = self.mask_correlated_clusters(self.class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.u = u
        self.temperature = 0.5

        self.xentropy = nn.CrossEntropyLoss().cuda()
        self.gama = torch.exp(torch.tensor([1.0])).cuda()
        self.kpa = 0.5

        self.wo1 = wo1
        self.wo2 = wo2
        self.wo3 = wo3

    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

    def cluster_contrastive_loss(self, c_i, c_j):
        c_i = c_i.t()
        c_j = c_j.t()
        N = 2 * self.class_num
        c = torch.cat((c_i, c_j), dim=0)

        sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_clusters = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        contrastive_loss = self.criterion(logits, labels)
        contrastive_loss /= N
        return contrastive_loss

    def forward(self, anchors, neighbors, epoch, epoch_num):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax

        b, c = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)

        # w_ij
        anchors_nl = anchors / torch.norm(anchors, dim=1).view(-1, 1).expand_as(anchors)
        neighbors_nl = neighbors / torch.norm(neighbors, dim=1).view(-1, 1).expand_as(neighbors)
        w = torch.exp(-torch.norm(anchors_nl - neighbors_nl, dim=1) ** 2 / self.tau_c)

        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, c), positives_prob.view(b, c, 1)).squeeze()

        if epoch < 130:
            consistency_loss = -torch.sum(torch.log(similarity), dim=0) / b
        else:
            consistency_loss = -torch.sum(w * torch.log(similarity), dim=0) / b  # add weight

        # cluster contrastive loss
        # contrastive_loss = self.cluster_contrastive_loss(anchors_prob, positives_prob)

        # gcc ccl one-hot
        # similarity = torch.mm(F.normalize(anchors_prob.t(), p=2, dim=1), F.normalize(positives_prob, p=2, dim=0))
        # contrastive_loss = self.xentropy(similarity, torch.arange(similarity.size(0)).cuda())


        # cluster contrastive loss with structure-guided
        similarity = torch.exp(torch.mm(F.normalize(anchors_prob.t(), p=2, dim=1), F.normalize(positives_prob, p=2, dim=0)))
        numerator = torch.diag(similarity)
        self.gama = (self.kpa * copy.copy(self.gama) + (1-self.kpa) * similarity) / torch.exp(torch.tensor([1.0])).cuda()
        similarity = self.gama*(similarity - torch.diag(numerator))
        contrastive_loss = - torch.sum( torch.log(numerator / torch.sum(similarity, dim=1))) / c

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)
        # entropy_loss = torch.sum(torch.norm(anchors_prob, dim=1)**2)/b


        # Total loss
        total_loss = self.wo1 * consistency_loss + self.wo2 * self.u * contrastive_loss - self.wo3 * self.entropy_weight*entropy_loss

        return total_loss, self.wo1 * consistency_loss, self.wo2 * self.u * contrastive_loss, self.wo3 * self.entropy_weight*entropy_loss


# similarity = torch.cat([similarity_p.unsqueeze(dim=1), similarity_n], dim=1)
# con_loss = nn.CrossEntropyLoss()
# target = torch.zeros(anchors.shape[0], dtype=torch.long, device=similarity.device)
# consistency_loss = con_loss(similarity, target)

class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]

        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert (n == 2)
        mask = torch.eye(b, dtype=torch.float32).cuda()

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature

        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).cuda(), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss

