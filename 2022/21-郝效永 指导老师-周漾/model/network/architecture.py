import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.spectral_norm as spectral_norm
from model.network.normalizition import PositionalNorm2d
from model.network.normalizition import SPADE
from utils.util import vgg_preprocess


class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size =3,padding =1 ,stride =1):
        super(ResidualBlock,self).__init__()
        self.relu = nn.PReLU()
        self.model = nn.Sequential(
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,padding=0,stride=stride),
            nn.InstanceNorm2d(out_channels),
            self.relu,
            nn.ReflectionPad2d(padding),
            nn.Conv2d(out_channels,out_channels,kernel_size=kernel_size,padding=0,stride=stride),
            nn.InstanceNorm2d(out_channels)
        )
    
    def forward(self,x):
        out = self.relu(x + self.model(x))
        return out
        pass

class SPADEResnetBlock(nn.Module):
    def __init__(self,fin,fout,opt=None,use_se = False,dilation = 1,ic =3,normalization = "spectral"):
        super().__init__()
        #Atribute
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin,fout)
        self.opt = opt
        self.pad_type = 'nozero'
        self.use_se = use_se
        self.normalization = normalization
        #creat conv layers
        if self.pad_type != 'zero':
            self.pad = nn.ReflectionPad2d(dilation)
            self.conv0 = nn.Conv2d(fin,fmiddle,kernel_size=3,padding=0,dilation=dilation)
            self.conv1 = nn.Conv2d(fmiddle,fout,kernel_size=3,padding=0,dilation=dilation)
        else:
            self.conv0 = nn.Conv2d(fin,fmiddle,kernel_size=3,padding=dilation,dilation=dilation)
            self.conv1 = nn.Conv2d(fmiddle,fout,kernel_size=3,padding=dilation,dilation=dilation)

        if self.learned_shortcut:
            self.convs = nn.Conv2d(fin,fout,kernel_size=1,bias=False)
        #apply spectral normalization if specified
        if self.normalization == "spectral":
            self.conv0 = spectral_norm(self.conv0)
            self.conv1 = spectral_norm(self.conv1)
            
            if self.learned_shortcut:
                self.convs = spectral_norm(self.convs)
        #define normalization layers
        spade_config_str = 'instance'

        self.ic = ic  ###  input channels of refmap ----label_nc

        self.norm_0 = SPADE(spade_config_str,fin,self.ic,PONO=True)
        self.norm_1 = SPADE(spade_config_str,fmiddle,self.ic,PONO=True)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str,fin,self.ic,PONO=True)
            
    def forward(self,x,seg1):#now ---- we use SPADE in U-net   ******  AND whats seg1???
        x_s = self.shortcut(x,seg1)
        if self.normalization == "positional":
            x_s = PositionalNorm2d(x_s)
        #x_s = self.shortcut_withoutSPADE(x)
        dx = self.conv0(self.pad(self.actvn(self.norm_0(x,seg1))))
        if self.normalization == "positional":
            dx = PositionalNorm2d(dx)
        dx = self.conv1(self.pad(self.actvn(self.norm_1(dx,seg1))))
        if self.normalization == "positional":
            dx = PositionalNorm2d(dx)
        out = x_s+dx
        return out

    def shortcut(self,x,seg1):
        if self.learned_shortcut:
            x_s = self.convs(self.norm_s(x,seg1))
        else:
            x_s = x
        return x_s
    # def shortcut_withoutSPADE(self,x):
    #     if self.learned_shortcut:
    #         x_s = self.convs(spectral_norm(x))
    #     else:
    #         x_s = x
    #     return x_s
    def actvn(self,x):
        return F.leaky_relu(x,2e-1)

class VGG_Feature_extractor(nn.Module):
    """
    NOTE: input tensor should range in [0,1]
    """
    def __init__(self,pool='max',vgg_normal_correct = False,ic = 3):
        super(VGG_Feature_extractor,self).__init__()
        self.vgg_normal_correct = vgg_normal_correct

        self.conv1_1 = nn.Conv2d(ic,64,kernel_size=3,padding=1)
        self.conv1_2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.conv2_1 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv2_2 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.conv3_1 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.conv3_2 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv3_3 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv3_4 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv4_1 = nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.conv4_2 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv4_3 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv4_4 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv5_1 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv5_2 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv5_3 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv5_4 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2,stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2,stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2,stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2,stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2,stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2,stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2,stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2,stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2,stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2,stride=2)
    def forward(self,x,out_keys,preprocess = True):
        """
        NOTE: input tensor should range in [0,1]
        """
        out = {}
        if preprocess:
            x = vgg_preprocess(x,vgg_normal_correct =self.vgg_normal_correct)
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))

        return [out[key] for key in out_keys]
        pass