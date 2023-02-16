"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8
import math
import numpy as np



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


class GATLoss(nn.Module):
    def __init__(self, num_classes, entropy_weight=5.0):
        super(GATLoss, self).__init__()
        self.softmax = nn.Softmax(dim=1)

        self.entropy_weight = entropy_weight  # Default = 2.0
        self.class_num = num_classes

        self.mask = self.mask_correlated_clusters(self.class_num)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

        self.temperature = 0.5


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

    def forward(self, gat_output):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]

        output:
            - Loss
        """
        # Softmax

        b, c = gat_output.size()
        gat_output_prob = self.softmax(gat_output)

        # cluster contrastive loss
        contrastive_loss = self.cluster_contrastive_loss(gat_output_prob, gat_output_prob)

        # Entropy loss
        entropy_loss = entropy(torch.mean(gat_output_prob, 0), input_as_probabilities=True)

        # Total loss
        total_loss = contrastive_loss - self.entropy_weight * entropy_loss

        return total_loss, contrastive_loss, entropy_loss



