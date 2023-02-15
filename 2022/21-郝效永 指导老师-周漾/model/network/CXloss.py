import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.util import feature_normalize

class ContextualLoss_forward(nn.Module):
    """
    input is A1 , B1 , channels = 1, range ~ [0 , 255]
    """
    def __init__(self):
        super(ContextualLoss_forward,self).__init__()
        return None
    def forward(self,x_features,y_features,h=0.1,feature_centering=True):
        batch_size = x_features.shape[0]
        feature_depth = x_features.shape[1]
        feature_size  = x_features.shape[2]

        #normalized feature vectors
        if feature_centering:
            x_features = x_features - y_features.mean(dim = 1).unsqueeze(dim=1)
            y_features = y_features - y_features.mean(dim = 1).unsqueeze(dim=1)
        x_features = feature_normalize(x_features).view(batch_size,feature_depth,-1)
        y_features = feature_normalize(y_features).view(batch_size,feature_depth,-1)

        x_features_permute = x_features.permute(0,2,1)
        d = 1 - torch.matmul(x_features_permute,y_features)

        d_norm = d / (torch.min(d,dim=-1,keepdim=True)[0] + 1e-3)

        w = torch.exp((1 - d_norm)/h)
        A_ij = w/torch.sum(w,dim=-1,keepdim=True)

        CX = torch.mean(torch.max(A_ij,dim=-1)[0],dim=1)
        loss = -torch.log(CX)

        return loss

        pass