import torch
import torch.nn as nn
import torch.nn.functional as F

from model.network.BaseModel import BaseNetwork
from model.network.normalizition import get_nospade_norm_layer
import utils.util as util

class MultiscaleDiscriminator(BaseNetwork):
    def __init__(self):
        super().__init__()
        num_D = 2
        for i in range(num_D):
            subnetD = self.create_single_discriminator()
            self.add_module('discriminator_%d' %i,subnetD)
    def create_single_discriminator(self):
        subarch = 'n_layer'
        netD = NLayerDiscriminator()
        return netD
        pass
    def downsample(self,input):
        return F.avg_pool2d(input,kernel_size = 3,stride=2,padding = [1,1] , count_include_pad = False)
    def forward(self,input):
        result = []
        get_intermediate_features = not True
        for name,D in self.named_children():
            out = D(input)
            if not get_intermediate_features:
                out = [out]
            result.append(out)
            input = self.downsample(input)
        return result

class NLayerDiscriminator(BaseNetwork):
    def __init__(self):
        super().__init__()
        kw = 4
        padw = int((kw - 1.0) / 2)
        nf = 64
        input_nc = self.compute_D_input_nc()
        label_nc = 20
        norm_layer = get_nospade_norm_layer('spectralinstance')
        sequence = [
            [
                nn.Conv2d(input_nc,nf,kernel_size=kw,stride=2,padding=padw),
                nn.LeakyReLU(0.2,False)
            ]
        ]
        for n in range(1,4):
            nf_prev = nf
            nf = min(nf * 2 ,512)
            stride = 1 if n == 3 else 2
            if n == 3:
                dec = []
                nc_dec = nf_prev
                for _ in range(3):
                    dec += [
                        nn.Upsample(scale_factor=2),norm_layer(nn.Conv2d(nc_dec,int(nc_dec//2),kernel_size=3,stride=1,padding=1)),
                        nn.LeakyReLU(0.2,False)
                    ]
                    nc_dec = int(nc_dec//2)
                dec += [nn.Conv2d(nc_dec,20,kernel_size=3,stride=1,padding=1)]
                self.dec = nn.Sequential(*dec)
            sequence += [
                [norm_layer(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=stride, padding=padw)),
                          nn.LeakyReLU(0.2, False)]
            ]
        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]
        for n in range(len(sequence)):
            self.add_module('model' + str(n), nn.Sequential(*sequence[n]))

    def compute_D_input_nc(self):
        input_nc = 3 + 20
        return input_nc
        pass
    def forward(self , input):
        results = [input]
        seg = None
        cam_logit = None
        for name,submodel in self.named_children():
            if 'model' not in name:
                continue
            x = results[-1]
            intermediate_output = submodel(x)
            results.append(intermediate_output)
        get_intermediate_features = not True
        if get_intermediate_features:
            retu = results[1:]
        else:
            retu = results[-1]
        return retu
        pass