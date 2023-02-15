import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from model.network.BaseModel import BaseNetwork
from model.network.architecture import SPADEResnetBlock

class SPADEGenerator(BaseNetwork):
    
    def __init__(self):
        super().__init__()
        num_of_nf = 8
        nf = 64
        label_nc = 20
        self.sw,self.sh = self.compute_latent_vector_size()
        ic = 4*3 + label_nc
        self.pre_MLP = nn.Conv2d(3 , label_nc, 1, padding= 0)
        self.fc = nn.Conv2d(ic , num_of_nf * nf , 3 , padding = 1) #why 8???
        self.head_0 = SPADEResnetBlock(num_of_nf*nf,num_of_nf*nf,ic = ic,normalization="positional")
        self.G_middle_0 = SPADEResnetBlock(num_of_nf * nf , num_of_nf * nf , ic = ic,normalization="positional")
        self.G_middle_1 = SPADEResnetBlock(num_of_nf * nf , num_of_nf * nf , ic = ic,normalization="positional")
        self.up_0 = SPADEResnetBlock(num_of_nf * nf, num_of_nf * nf , ic = ic,normalization="positional")
        self.up_1 = SPADEResnetBlock(num_of_nf * nf , num_of_nf // 2 * nf , ic = ic,normalization="positional")
        self.up_2 = SPADEResnetBlock(num_of_nf // 2 * nf , num_of_nf // 4 * nf , ic = ic,normalization="positional")
        self.up_3 = SPADEResnetBlock(num_of_nf // 4 * nf , num_of_nf // 8 * nf , ic = ic,normalization="positional")
        final_nc = nf
        self.conv_img = nn.Conv2d(final_nc,3,3,padding=1)
        self.up = nn.Upsample(scale_factor=2)
        pass

    def compute_latent_vector_size(self):
        num_up_layers = 5
        sw = 128 // (2**num_up_layers)
        sh = round(sw/1.0)
        return sw,sh
        pass
    """ whats input """
    def forward(self, input , warp_out = None):
        #input = self.pre_MLP(input) # 3 x 128 x 128 -> 182 x 128 x 128
        seg = torch.cat((F.interpolate(warp_out[0],size=(128,128)) , F.interpolate(warp_out[1],size=(128,128)) , F.interpolate(warp_out[2],size=(128,128)) , warp_out[3] , input) , dim=1)
        x = F.interpolate(seg ,size=(self.sh,self.sw)) # 194 x 8 x 8
        x = self.fc(x) # 194 x 4 x 4 -> 512 x 4 x 4
        x = self.head_0(x,seg) # 512 x 4 x 4 -> 512 x 4 x 4
        x = self.up(x) # 512 x 8 x 8
        x = self.G_middle_0(x,seg) #512 x 8 x 8
        x = self.G_middle_1(x,seg) #512 x 8 x 8
        x = self.up(x) # 512 x 16 x 16
        x = self.up_0(x , seg) # 512 x 16 x 16
        x = self.up(x) # 512 x 32 x 32
        x = self.up_1(x , seg) # 256 x 32 x 32
        x = self.up(x) # 256 x 64 x 64
        x = self.up_2(x , seg) # 128 x 64 x 64
        x = self.up(x) # 128 x 128 x128
        x = self.up_3(x , seg) # 64 x 128 x 128
        x = self.conv_img(F.leaky_relu(x,2e-1))
        x = torch.tanh(x)
        return x
        pass