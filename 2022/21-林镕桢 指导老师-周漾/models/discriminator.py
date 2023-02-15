if __name__ == "__main__":
    import sys
    sys.path.append("/home/ubuntu/my_spade")

from args.model_args import get_model_args
from torch.nn.utils.spectral_norm import spectral_norm
import torch

class DownSampleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c, stride=2) -> None:
        super().__init__()
        self.conv = spectral_norm(torch.nn.Conv2d(in_c, out_c, 4, stride, 1))
        self.norm = torch.nn.InstanceNorm2d(out_c)
    
    def forward(self, x):
        return self.norm(self.conv(x))


class Discriminator(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.in_c = args.img_c + args.seg_c
        self.start_c = args.start_c
        self.act_func = torch.nn.LeakyReLU(0.2)
        self.conv_in = torch.nn.Conv2d(self.in_c, self.start_c, 4, 2, 1)

        self.downsample_list = torch.nn.ModuleList()
        self.downsample_list.append(DownSampleBlock(self.start_c, self.start_c*2))
        self.downsample_list.append(DownSampleBlock(self.start_c*2, self.start_c*4))
        self.downsample_list.append(DownSampleBlock(self.start_c*4, self.start_c*8, 1))

        self.conv_out = torch.nn.Conv2d(self.start_c*8, 1, 4)

    def forward(self, x):
        output = self.act_func(self.conv_in(x))
        res = [output]
        for i in range(3):
            output = self.downsample_list[i](output)
            output = self.act_func(output)
            res.append(output)
        output = self.conv_out(output)
        res.append(output)
        return res


if __name__ == "__main__":
    args = get_model_args()

    x = torch.randn(1, 187, 256, 256)
    discrim = Discriminator(args)
    res = discrim(x)

    print(res[-1].shape)