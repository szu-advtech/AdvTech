import torch
from torch import nn
from torchvision.models.vgg import vgg16
import numpy as np
import torch.nn.functional as F

class DiscriminateLoss(nn.Module):
    def __init__(self):
        super(DiscriminateLoss, self).__init__()
        self.L1_loss = nn.MSELoss()

    def forward(self, predict_labels, target_labels):
        # Image Label Loss
        label_loss = self.L1_loss(predict_labels, target_labels)
        return label_loss

class rankLoss(nn.Module):
    def __init__(self):
        super(rankLoss, self).__init__()
        self.L1_loss = nn.MSELoss()

    def forward(self, predict_labels1,predict_labels2, target_labels1, target_labels2):
        # Image Label Loss
        label_loss = self.L1_loss(predict_labels1- predict_labels2, target_labels1- target_labels2)
        return label_loss

class type_strenghloss(nn.Module):
    def __init__(self):
        super(type_strenghloss, self).__init__()
        self.crossentropyloss = nn.CrossEntropyLoss()

    def forward(self, predict_labels, target_labels):
        # Image Label Loss
        CrossEntropy_Loss = self.crossentropyloss(predict_labels, target_labels)
        batch_size = predict_labels.size(0)
        _, pred = predict_labels.max(1)
        num_correct = (pred == target_labels).sum().item()
        accuracy = num_correct / batch_size
        return CrossEntropy_Loss, accuracy


if __name__ == "__main__":
    g_loss = DiscriminateLoss()
    print(g_loss)
