import re

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm

def get_nospade_norm_layer(norm_type = 'instance'):
    def get_out_channel(layer):
        if hasattr(layer,'out_channels'):
            return getattr(layer,'out_channels')
        return layer.weight_size(0)
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith('spectral'):
            layer = spectral_norm(layer)
            subnorm_type = norm_type[len('spectral'):]
        else:
            subnorm_type = norm_type
        if subnorm_type == 'none' or len(subnorm_type) == 0:
            return layer

        if getattr(layer,'bias',None) is not None:
            delattr(layer,'bias')
            layer.register_parameter('bias',None)
        if subnorm_type == 'batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer),affine=True)
        elif subnorm_type == 'sync_batch':
            norm_layer = nn.BatchNorm2d(get_out_channel(layer),affine=True)
        elif subnorm_type == 'instance':
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer),affine=False)
        else:
            raise ValueError('normalization layer %s is not recognized' % subnorm_type)
        return nn.Sequential(layer,norm_layer)
    return add_norm_layer
    pass

class SPADE(nn.Module): #norm_nc --target channels   label_nc --label input channels
    def __init__(self,config_text,norm_nc,label_nc,PONO=False):
        super().__init__()
        param_free_norm_type = 'instance'
        ks = 3
        self.pad_type = 'nozero'
        self.param_free_norm = PositionalNorm2d
        nhidden = 128
        pw = ks//2
        
        self.mlp_shared = nn.Sequential(
                nn.ReflectionPad2d(pw),
                nn.Conv2d(label_nc,nhidden,kernel_size=ks,padding=0),
                nn.ReLU()
            )
        self.pad = nn.ReflectionPad2d(pw)
        self.mlp_gamma = nn.Conv2d(nhidden,norm_nc,kernel_size=ks,padding=0)
        self.mlp_beta = nn.Conv2d(nhidden,norm_nc,kernel_size=ks,padding=0)

    def forward(self,x,segmap):
        normalized =  self.param_free_norm(x)
        segmap = F.interpolate(segmap,size=x.size()[2:],mode='nearest')
        actv = self.mlp_shared(segmap)
        if self.pad_type != 'zero':
            gamma = self.mlp_gamma(self.pad(actv))
            beta = self.mlp_beta(self.pad(actv))
        else:
            gamma = self.mlp_gamma(actv)
            beta = self.mlp_beta(actv)
        out = normalized * (1+gamma) + beta
        return out
    pass

def PositionalNorm2d(x,epsion=1e-8):
    #x : B*C*W*H normalize in C dim
    mean = x.mean(dim=1,keepdim=True)
    std = x.var(dim =1,keepdim=True).add(epsion).sqrt()
    output = (x - mean)/std
    return output
    pass