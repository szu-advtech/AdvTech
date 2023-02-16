import torch
import torch.nn as nn
from torch.nn import functional as F

class Projector(nn.Module):
    def __init__(self):
        super(Projector, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(2048 + 128, 2048 + 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2048 + 128, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024)

        )
    def forward(self,x):
        return self.layer(x)

