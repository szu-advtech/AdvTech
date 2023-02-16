import torch
import torch.nn as nn
from .Vgg_16 import Vgg_face,Mask_net
from torchsummary import summary

class EFE(nn.Module):
    def __init__(self,pretrained = True):
        super(EFE, self).__init__()
        if pretrained:
            vgg_16 = Vgg_face()
            vgg_16.load_state_dict(torch.load("/data/qzhuang/Projects/Task1/model/vgg.pth"))
        else:
            vgg_16 = Vgg_face()

        self.EFE_block_1 = torch.nn.Sequential(*(list(vgg_16.children())[:-7]))
        self.EFE_block_2 = Mask_net()

        # Fc Layer
        self.EFE_classify = nn.Linear(in_features=25088,out_features=7,bias=True)

        # init
        # nn.init.uniform_(self.EFE_classify.weight,a=-0.1,b=0.1)
        # nn.init.constant_(self.EFE_classify.bias,0.1)

        #if not pretrained:

    def forward(self,image):
        Gori = self.EFE_block_1(image)
        mask = self.EFE_block_2(Gori)
        # Gmask is not the same as Gori address, The address is different after the calculation
        Gmask = mask * Gori
        # Gmask_reshape = F.avg_pool2d(Gmask, Gmask.size()[2:])
        Gmask_reshape = Gmask.view(Gmask.shape[0],-1)
        # baseline
        pre_EFE = self.EFE_classify(Gmask_reshape)
        # pre_EFE = self.EFE_classify(Gori.view(Gori.shape[0],-1))

        return  mask ,Gmask ,pre_EFE











