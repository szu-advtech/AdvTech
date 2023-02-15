if __name__ == "__main__":
    import sys
    sys.path.append("/home/ubuntu/my_spade")

import torch
from args.model_args import get_model_args

class DownSampleBlock(torch.nn.Module):
    def __init__(self, in_c, out_c) -> None:
        super().__init__()
        self.conv = torch.nn.Conv2d(in_c, out_c, 3, 2, 1)
        self.norm = torch.nn.InstanceNorm2d(out_c)
    
    def forward(self, x):
        return self.norm(self.conv(x))


class Encoder(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        in_c = args.img_c
        start_c = args.start_c
        img_size = args.crop_size
        self.scale_time = args.en_downsample_times

        self.in_conv = torch.nn.Conv2d(in_c, start_c, 3, 2, 1)
        self.act_func = torch.nn.LeakyReLU(0.2, False)

        self.downsample_list = torch.nn.ModuleList()
        self.downsample_list.append(DownSampleBlock(start_c, start_c*2))
        self.downsample_list.append(DownSampleBlock(start_c*2, start_c*4))
        self.downsample_list.append(DownSampleBlock(start_c*4, start_c*8))
        self.downsample_list.append(DownSampleBlock(start_c*8, start_c*8))
        self.downsample_list.append(DownSampleBlock(start_c*8, start_c*8))

        self.mu = torch.nn.Linear(512*((img_size//(2**self.scale_time))**2), 256)
        self.sigma = torch.nn.Linear(512*((img_size//(2**self.scale_time))**2), 256)

    def forward(self, x):
        output = self.in_conv(x)
        for i in range(self.scale_time-1):
            output = self.downsample_list[i](output)
            output = self.act_func(output)

        output = output.view(x.shape[0],-1)
        mu = self.mu(output)
        log_s = self.sigma(output)

        return mu, log_s


if __name__ == "__main__":
    args = get_model_args()

    x = torch.randn(1, 3, 256, 256)
    discrim = Encoder(x.shape[1], args)
    res = discrim(x)

    print(res[0].shape, res[1].shape)