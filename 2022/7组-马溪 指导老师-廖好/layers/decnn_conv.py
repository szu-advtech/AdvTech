import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DECNN_CONV(nn.Module):
    def __init__(self, input_dim, opt):
        super(DECNN_CONV, self).__init__()

        self.conv1 = torch.nn.Conv1d(input_dim, 128, 5, padding=2)
        self.conv2 = torch.nn.Conv1d(input_dim, 128, 3, padding=1)
        self.dropout = torch.nn.Dropout(opt.keep_prob)

        self.conv3 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv4 = torch.nn.Conv1d(256, 256, 5, padding=2)
        self.conv5 = torch.nn.Conv1d(256, 256, 5, padding=2)

    def forward(self, inputs):
        x_conv = torch.nn.functional.relu(torch.cat((self.conv1(inputs), self.conv2(inputs)), dim=1))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv5(x_conv))
        x_conv = x_conv.transpose(1, 2)
        return x_conv