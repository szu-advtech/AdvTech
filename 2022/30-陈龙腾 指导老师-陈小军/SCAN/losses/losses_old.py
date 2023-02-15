"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
EPS=1e-8
import math
import numpy as np


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
        return F.cross_entropy(input, target, weight = weight, reduction = reduction)


class ConfidenceBasedCE(nn.Module):
    def __init__(self, threshold, apply_class_balancing):
        super(ConfidenceBasedCE, self).__init__()
        self.loss = MaskedCrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.logsoftmax_func = nn.LogSoftmax(dim=1)
        self.threshold = threshold
        self.apply_class_balancing = apply_class_balancing

    def forward(self, anchors_weak, anchors_strong, neighbors, ct, h):
        """
        Loss function during self-labeling

        input: logits for original samples and for its strong augmentations
        output: cross entropy
        """
        self.threshold = ct

        # Retrieve target and mask based on weakly augmentated anchors
        weak_anchors_prob = self.softmax(anchors_weak)
        b, c = weak_anchors_prob.size()
        max_prob, target = torch.max(weak_anchors_prob, dim=1)
        mask = max_prob > self.threshold
        target_masked = torch.masked_select(target, mask.squeeze())
        n = target_masked.size(0)

        # confidence
        neighbors_prob = neighbors.softmax(2)
        anchors_weak_nl = anchors_weak / torch.norm(anchors_weak, dim=1).view(-1, 1).expand_as(anchors_weak)
        neighbors_nl = neighbors / torch.norm(neighbors, dim=2).view(b, -1, 1).expand_as(neighbors)
        beta = torch.sum(torch.exp(-torch.norm(anchors_weak_nl.unsqueeze(1) - neighbors_nl, dim=2) ** 2).unsqueeze(
            2) * neighbors_prob, dim=1)
        beta = beta / beta.sum(1).view(-1, 1).expand_as(beta)
        alpha_ = torch.max(torch.ones_like(max_prob), torch.norm(weak_anchors_prob - beta, dim=1) ** (-1))
        alpha = torch.clamp(alpha_, max=100)
        q = []
        for i in range(len(alpha)):
            qi = weak_anchors_prob[i] ** alpha[i]
            qi = qi / qi.sum(0)
            q.append(qi.unsqueeze(0))
        q = torch.cat(q, dim=0)

        # Inputs are strongly augmented anchors
        input_ = anchors_strong

        # Class balancing weights
        if self.apply_class_balancing:
            idx, counts = torch.unique(target_masked, return_counts=True)
            # freq = 1 / (counts.float() / n)
            weight_ = torch.ones(c).cuda()
            # weight[idx] = freq

            freq = counts.float() / n
            # h = 1.02
            weight_[idx] = 1/torch.log(h+freq)
            weight = torch.clamp(weight_, min=1, max=50)

        else:
            weight = None

        # Loss
        n = torch.sum(mask > 0)
        input = torch.masked_select(anchors_strong, mask.view(b, 1)).view(n, c)
        input_prob = self.logsoftmax_func(input)
        q_mask = torch.masked_select(q, mask.view(b, 1)).view(n, c)

        # loss = self.loss(input_, target, mask, weight = weight, reduction='mean')
        w_avg = weight.view(1, -1) / torch.sum(weight) * torch.mean(weight)
        loss = -torch.sum(torch.sum(w_avg*q_mask * input_prob, dim=1) , dim=0) / n  # add weight
        return loss


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

def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 

    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = EPS)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, u,num_classes, entropy_weight = 5.0, weight_t = 1.0, alpha = 0.1):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = None
        self.entropy_weight = entropy_weight # Default = 2.0
        self.weight_t = weight_t
        self.alpha =alpha
        self.class_num = num_classes

        self.mask = self.mask_correlated_clusters(self.class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)
        self.u = u
        self.temperature = 1.0



    def mask_correlated_clusters(self, class_num):
        N = 2 * class_num
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(class_num):
            mask[i, class_num + i] = 0
            mask[class_num + i, i] = 0
        mask = mask.bool()
        return mask

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
        # positives_prob = self.softmax(neighbors)
        positives_prob = neighbors.softmax(2)

        # w_ij
        anchors_nl = anchors / torch.norm(anchors, dim=1).view(-1, 1).expand_as(anchors)
        neighbors_nl = neighbors / torch.norm(neighbors, dim=2).view(b, -1, 1).expand_as(neighbors)
        t = 1.0
        w = torch.exp(-torch.norm(anchors_nl.unsqueeze(1) - neighbors_nl, dim=2) ** 2 / t)

        # Similarity in output space
        # similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        # consistency_loss = -torch.sum(omega*torch.log(similarity))/torch.sum(omega)
        similarity = torch.einsum('ik,ihk->ih', anchors_prob, positives_prob)
        d1, d2 = similarity.shape
        if epoch < 15:
            consistency_loss = -torch.sum(torch.sum(torch.log(similarity), dim=1) , dim=0) / (d1*d2)
        else:
            consistency_loss = -torch.sum(torch.sum(w*torch.log(similarity), dim=1), dim=0) / (d1 * d2)  # add weight


        # cluster contrastive loss
        # c_i = anchors_prob
        # c_j = positives_prob
        # c_i = c_i.t()
        # c_j = c_j.t()
        # N = 2 * self.class_num
        # c = torch.cat((c_i, c_j), dim=0)
        #
        # sim = self.similarity_f(c.unsqueeze(1), c.unsqueeze(0)) / self.temperature
        # sim_i_j = torch.diag(sim, self.class_num)
        # sim_j_i = torch.diag(sim, -self.class_num)
        #
        # positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        # negative_clusters = sim[self.mask].reshape(N, -1)
        #
        # labels = torch.zeros(N).to(positive_clusters.device).long()
        # logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        # contrastive_loss = self.criterion(logits, labels)
        # contrastive_loss /= N

        contrastive_loss = torch.zeros_like(consistency_loss)
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss


        return total_loss, consistency_loss, contrastive_loss, entropy_loss


class CCLoss(nn.Module):
    def __init__(self, entropy_weight=2.0, weight_t=1.0, alpha=1.0):
        super(CCLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight  # Default = 2.0
        self.weight_t = weight_t
        self.alpha = alpha

    def forward(self, anchors, neighbors, disk):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """

        # positives_prob = neighbors.softmax(dim=2)  # for all knn simples
        # similarity_p = torch.einsum('ij,ikj->ik', anchors_prob, positives_prob)  # for all knn simples

        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)  # for randomly one knn sample
        disk_prob = disk.softmax(dim=2)

        similarity_p = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()  # for randomly one knn sample
        pos_similarity = torch.sum(torch.log(similarity_p))

        similarity_n = torch.einsum('ij,ikj->ik', anchors_prob,disk_prob)
        neg_similarity = torch.sum(torch.log(torch.sum(torch.exp(similarity_n), dim=1)))

        eta = 1.0/disk_prob.shape[1]*self.alpha  # (n+)/(n-)*alpha
        consistency_loss = (-pos_similarity+neg_similarity)/b
        # consistency_loss = (-pos_similarity)/b

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return total_loss, consistency_loss, entropy_loss


# based on the samples lower than the averaged similarities
class CCLoss2(nn.Module):
    def __init__(self, entropy_weight=2.0, weight_t=1.0, alpha=1.0):
        super(CCLoss2, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight  # Default = 2.0
        self.weight_t = weight_t
        self.alpha = alpha

    def forward(self, anchors, neighbors, disks, epoch, t):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """

        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)  # for randomly one knn sample
        disks_prob = disks.softmax(dim=2)

        similarity_p = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()  # for randomly one knn sample
        similarity_n = torch.einsum('ij,ikj->ik', anchors_prob, disks_prob)  # 128*10   128*dk*10

        # add weight
        # norm_ij = torch.norm(anchors_prob-positives_prob, dim=1)**2
        # p_weight = 1.0 / torch.exp(norm_ij/t)
        #
        # norm_ij2 = torch.norm(anchors_prob.unsqueeze(1)-disks_prob, dim=2)**2
        # n_weight = 1.0 / torch.exp(norm_ij2 / t)
        #
        # similarity_p = similarity_p*p_weight
        # similarity_n = similarity_n*n_weight


        ones = torch.ones_like(similarity_p)

        pos_loss = self.bce(similarity_p, ones)
        neg_loss = torch.sum(torch.log(1 + torch.sum(similarity_n, dim=1)))

        eta = 1.0 / similarity_n.shape[1]  # (n+)/(n-)*alpha
        if epoch < 101:
            eta = 1.0

        consistency_loss = pos_loss + eta*neg_loss/b

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return pos_loss, eta*neg_loss/b, total_loss, consistency_loss, entropy_loss


# similarity = torch.cat([similarity_p.unsqueeze(dim=1), similarity_n], dim=1)
# con_loss = nn.CrossEntropyLoss()
# target = torch.zeros(anchors.shape[0], dtype=torch.long, device=similarity.device)
# consistency_loss = con_loss(similarity, target)


class CCLoss3(nn.Module):
    def __init__(self, entropy_weight=2.0, weight_t=1.0, alpha=1.0):
        super(CCLoss3, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight  # Default = 2.0
        self.weight_t = weight_t
        self.alpha = alpha

    def forward(self, anchors, neighbors, disks, epoch):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """

        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)  # for randomly one knn sample
        disks_prob = disks.softmax(dim=2)

        similarity_p = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()  # for randomly one knn sample
        similarity_n = torch.einsum('ij,ikj->ik', anchors_prob, disks_prob)  # 128*10   128*dk*10

        # add weight
        t = 1.5
        norm_ij = torch.norm(anchors_prob-positives_prob, dim=1)**2
        p_weight = 1.0 / torch.exp(norm_ij/t)

        norm_ij2 = torch.norm(anchors_prob.unsqueeze(1)-disks_prob, dim=2)**2
        n_weight = 1.0 / torch.exp(norm_ij2 / t)

        similarity_p = similarity_p*p_weight
        # similarity_n = similarity_n*n_weight

        ones = torch.ones_like(similarity_p)
        Pos_loss = self.bce(similarity_p, ones)

        Neg_loss = torch.sum(torch.log(1 + torch.sum(similarity_n, dim=1)))

        if epoch < 15:
            eta = 0
        else:
            eta = 1.0 / similarity_n.shape[1]*10  # (n+)/(n-)*alpha
            # eta = 1.0
        Neg_loss = eta * Neg_loss/b

        consistency_loss = Pos_loss + Neg_loss

        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss

        return Pos_loss, Neg_loss, total_loss, consistency_loss, entropy_loss





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
        assert(n == 2)
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
