import os
import torch
from torch.nn import init
from models.encoder import Encoder
from models.generator import Generator
from models.discriminator import Discriminator
from args.model_args import get_model_args


class Model(torch.nn.Module):
    def __init__(self, is_train=True) -> None:
        super().__init__()
        self.args = get_model_args()
        self.is_encode = self.args.is_encode
        self.is_clip = self.args.is_clip
        self.is_train = is_train

        self.net_g = Generator(self.args)
        if self.is_encode:
            self.encoder = Encoder(self.args)
        if self.is_train:
            self.net_d = Discriminator(self.args)

        self.init_weights('xavier')

    def encode_img(self, img):
        mu, log_s = self.encoder(img)
        return mu, log_s

    def get_nosie(self, mu, log_s):
        std = (0.5*log_s).exp()
        eps = torch.randn_like(std)
        z = eps.mul(std) + mu
        return z
        
    def gen_img(self, z, seg, feature=None):
        return self.net_g(z, seg, feature)

    def det_img(self, img, seg):
        img_seg = torch.cat([img, seg], dim=1)
        score = self.net_d(img_seg)
        return score

    def forward(self, seg, rank=0, img=None, clip_feature=None):
        if self.is_encode:
            if not img is None:
                mu, log_s = self.encode_img(img)
                z = self.get_nosie(mu, log_s)
            else:
                z = torch.randn([seg.shape[0], 256]).to(rank)
        else:
            size = self.net_g.start_size
            z = torch.nn.functional.interpolate(seg, (size, size))
            if clip_feature is not None:
                return self.gen_img(z, seg, clip_feature)

        return self.gen_img(z, seg)

    def save(self, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(self.net_g.state_dict(), save_path + "/net_g.pth")
        torch.save(self.net_d.state_dict(), save_path + "/net_d.pth")
        if self.is_encode:
            torch.save(self.encoder.state_dict(), save_path + "/encoder.pth")

    def load(self, save_path):
        self.net_g.load_state_dict(torch.load(save_path + "/net_g.pth"))
        if self.is_encode:
            self.encoder.load_state_dict(torch.load(save_path + "/encoder.pth"))
        if self.is_train:
            self.net_d.load_state_dict(torch.load(save_path + "/net_d.pth"))

    def init_weights(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm2d') != -1 or classname.find('SyncBatchNorm') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    init.normal_(m.weight.data, 1.0, gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'xavier_uniform':
                    init.xavier_uniform_(m.weight.data, gain=1.0)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                elif init_type == 'none':  # uses pytorch's default init method
                    m.reset_parameters()
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

        self.apply(init_func)
