
from torch import nn
from torchvision import models
import torch.nn.functional as F

from SSDA_MME.MME.loss import grad_reverse


class AlexNetBase(nn.Module):
    def __init__(self,pret=True):
        super(AlexNetBase, self).__init__()
        model_alexnet=models.alexnet(pretrained=pret)
        self.features=nn.Sequential(*list(model_alexnet.features._modules.values())[:])
        self.classifier=nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier"+str(i),model_alexnet.classifier[i])
        self.__in_features=model_alexnet.classifier[6].in_features

    def forward(self,x):
        x=self.features(x)
        x=x.view(x.size(0),256*6*6)
        x=self.classifier(x)
        return x

    def output_num(self):
        return self.__in_features


class Predictor(nn.Module):
    def __init__(self,num_class=31,inc=4096,temp=0.05):
        super(Predictor, self).__init__()
        self.fc=nn.Linear(inc,num_class,bias=False)
        self.num_class=num_class
        self.temp=temp

    def forward(self,x,reverse=False,eta=0.1):
        if reverse:
            x=grad_reverse(x,eta)
        x=F.normalize(x)
        x_out=self.fc(x)/self.temp
        return x_out

