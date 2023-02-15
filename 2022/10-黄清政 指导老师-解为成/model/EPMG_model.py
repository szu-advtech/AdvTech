import torch
import torch.nn as nn
from torchsummary import summary


class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=256, kernel_size=(4, 4), stride=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(4, 4), stride=(1, 1)),
            nn.Sigmoid(),
        )
        # for name,param  in self.features.named_parameters():
        #     if name.startswith("weight"):
        #         nn.init.normal_(param)
        #     else:
        #         nn.init.zeros_(param)
    def forward(self, Gmask):
        Gmask = self.features(Gmask)
        return Gmask

class Fc(nn.Module):
    def __init__(self, nums=7):
        super(Fc, self).__init__()
        self.nums = nums
        self.classifer = nn.Sequential(
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=self.nums),
            nn.Softmax(dim=1),
        )
        # for name,param  in self.classifer.named_parameters():
        #     if name.startswith("weight"):
        #         nn.init.uniform_(param,a=-0.1,b=0.1)
        #     else:
        #         nn.init.zeros_(param)
    def forward(self, E):
        E = self.classifer(E)

        return E


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(4, 4)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(4, 4)),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=8, out_channels=4, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=4, out_channels=2, kernel_size=(4, 4), stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=2, out_channels=1, kernel_size=(4, 4), stride=2, padding=1),
            nn.Sigmoid()
        )

        # 参数初始化
        # for name,param  in self.decoder.named_parameters():
        #     if name.startswith("weight"):
        #         nn.init.normal_(param)
        #     else:
        #         nn.init.zeros_(param)
    def forward(self, feature):
        feature = self.decoder(feature)
        return feature

class EPMG(nn.Module):
    def __init__(self):
        super(EPMG, self).__init__()
        self.backbone1 = Encoder()
        self.backbone2 = Fc()
        self.backbone3 = Decoder()

    def forward(self, feature):
        feature = self.backbone1(feature)
        e = feature.view(feature.shape[0], -1)
        c_e = self.backbone2(e)
        P = self.backbone3(feature)
        P = P.view(P.shape[0], 224, 224)
        return e, c_e, P

