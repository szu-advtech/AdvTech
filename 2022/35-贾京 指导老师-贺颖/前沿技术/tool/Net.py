
from torch import nn
import torch.nn.functional as F



class MlpMnist(nn.Module):
    def __init__(self):
        super(MlpMnist,self).__init__()
        self.fc1 = nn.Linear(28*28,80)
        self.fc2 = nn.Linear(80,60)
        self.fc3 = nn.Linear(60,10)

    def forward(self,x):
        x = x.view(-1,28*28)

        x = self.fc1(x)
        x = F.elu(x)

        x = self.fc2(x)
        x = F.elu(x)

        x = self.fc3(x)
        x = F.elu(x)

        return x



