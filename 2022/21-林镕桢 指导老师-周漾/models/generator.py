if __name__ == "__main__":
    import sys
    sys.path.append("/home/ubuntu/spade/my_spade_final")

import torch
from torch.nn.utils.spectral_norm import spectral_norm
from args.model_args import get_model_args


class SPADE(torch.nn.Module):
    def __init__(self, x_c, seg_c, hiden_c=128, is_clip=False) -> None:
        super().__init__()
        self.hiden_c = hiden_c
        self.is_clip = is_clip
        # self.normalize = torch.nn.BatchNorm2d(x_c, affine=False)
        self.normalize = torch.nn.SyncBatchNorm(x_c, affine=False)
        self.shared = torch.nn.Conv2d(seg_c, hiden_c, 3, 1, 1)
        self.act_func = torch.nn.ReLU(True)
        self.scale = torch.nn.Conv2d(hiden_c, x_c, 3, 1, 1)
        self.bias = torch.nn.Conv2d(hiden_c, x_c, 3, 1, 1)

        if self.is_clip:
            # self.clip_mean_0 = torch.nn.Linear(512, 184)
            # self.clip_mean_1 = torch.nn.Linear(512, hiden_c)
            # self.clip_std_0 = torch.nn.Linear(512, 184)
            # self.clip_std_1 = torch.nn.Linear(512, hiden_c)
            self.clip_mean = torch.nn.Sequential(
                torch.nn.Linear(512, 1024),
                torch.nn.Linear(1024, hiden_c),
            )
            
            self.clip_std = torch.nn.Sequential(
                torch.nn.Linear(512, 1024),
                torch.nn.Linear(1024, hiden_c),
            )

    def forward(self, x, seg_map, clip_feature=None):
        norm = self.normalize(x)
        seg_map = torch.nn.functional.interpolate(seg_map, x.size()[2:], mode="nearest")
        
        if self.is_clip:
            # mean = self.clip_mean_0(clip_feature).unsqueeze(-1).unsqueeze(-1)
            # std = self.clip_std_0(clip_feature).unsqueeze(-1).unsqueeze(-1)
            # output = (1+std)*seg_map + mean
            # output = self.shared(output)
            
            # mean = self.clip_mean_1(clip_feature).unsqueeze(-1).unsqueeze(-1)
            # std = self.clip_std_1(clip_feature).unsqueeze(-1).unsqueeze(-1)
            output = self.shared(seg_map)
            mean = self.clip_mean(clip_feature).unsqueeze(-1).unsqueeze(-1)
            std = self.clip_std(clip_feature).unsqueeze(-1).unsqueeze(-1)
            output = (1+std)*output + mean
        else:
            output = self.shared(seg_map)
        seg_act = self.act_func(output)
        scale = self.scale(seg_act)
        bias = self.bias(seg_act)
        output = norm * (1+scale) + bias

        return output


class SPADEResBlk(torch.nn.Module):
    def __init__(self, in_c, out_c, args) -> None:
        super().__init__()
        spade_hiden_c = args.spade_hiden_c
        mid_c = min(in_c, out_c)
        self.is_shortcut = not (in_c == out_c)
        self.is_clip = args.is_clip

        self.act_func = torch.nn.LeakyReLU(0.2, True)
        self.spade_0 = SPADE(in_c, args.seg_c, spade_hiden_c, self.is_clip)
        self.conv_0 = torch.nn.Conv2d(in_c, mid_c, 3, 1, 1)
        self.spade_1 = SPADE(mid_c, args.seg_c, spade_hiden_c, self.is_clip)
        self.conv_1 = torch.nn.Conv2d(mid_c, out_c, 3, 1, 1)
        if self.is_shortcut:
            self.spade_cut = SPADE(in_c, args.seg_c, spade_hiden_c)
            self.conv_cut = torch.nn.Conv2d(in_c, out_c, 3, 1, 1, bias=False)
        if args.spectral_norm:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.is_shortcut:
                self.conv_cut = spectral_norm(self.conv_cut)
    
    def forward(self, x, seg_map, clip_feature=None):
        output = self.spade_0(x, seg_map, clip_feature)
        output = self.act_func(output)
        output = self.conv_0(output)
        output = self.spade_1(output, seg_map, clip_feature)
        output = self.act_func(output)
        output = self.conv_1(output)
        if self.is_shortcut:
            x = self.conv_cut(self.spade_cut(x, seg_map))
        return output + x



class Generator(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        self.args = args
        self.is_clip = args.is_clip
        self.start_size = self.args.crop_size // (2**self.args.upsample_times)
        self.start_c = self.args.last_spade_c * (2**self.args.c_downsample_times)
        if self.args.is_encode:
            self.in_layer = torch.nn.Linear(self.args.z_c, self.start_c*(self.start_size**2))
        else:
            self.in_layer = torch.nn.Conv2d(self.args.seg_c, self.start_c, 3, 1, 1)

        if self.is_clip:
            # self.clip_mean_0 = torch.nn.Linear(512, 184)
            # self.clip_mean_1 = torch.nn.Linear(512, self.start_c)
            # self.clip_std_0 = torch.nn.Linear(512, 184)
            # self.clip_std_1 = torch.nn.Linear(512, self.start_c)
            self.clip_mean = torch.nn.Sequential(
                torch.nn.Linear(512, 1024),
                torch.nn.Linear(1024, self.start_c),
            )
            self.clip_std = torch.nn.Sequential(
                torch.nn.Linear(512, 1024),
                torch.nn.Linear(1024, self.start_c),
            )
        self.spade_list = torch.nn.ModuleList()
        self.spade_list.append(SPADEResBlk(self.start_c, self.start_c, self.args))
        self.spade_list.append(SPADEResBlk(self.start_c, self.start_c, self.args))
        self.spade_list.append(SPADEResBlk(self.start_c, self.start_c, self.args))
        self.spade_list.append(SPADEResBlk(self.start_c, self.start_c//2, self.args))
        self.spade_list.append(SPADEResBlk(self.start_c//2, self.start_c//4, self.args))
        self.spade_list.append(SPADEResBlk(self.start_c//4, self.start_c//8, self.args))
        self.spade_list.append(SPADEResBlk(self.start_c//8, self.start_c//16, self.args))


        self.upsample = torch.nn.Upsample(scale_factor=2)
        self.conv_out = torch.nn.Conv2d(self.args.last_spade_c, 3, 3, 1, 1)
        self.l_relu = torch.nn.LeakyReLU(0.2, True)
        self.act_func = torch.nn.Tanh()



    def forward(self, x, seg_map, clip_feature=None):
        if self.is_clip:
            # mean = self.clip_mean_0(clip_feature).unsqueeze(-1).unsqueeze(-1)
            # std = self.clip_std_0(clip_feature).unsqueeze(-1).unsqueeze(-1)
            # output = (1+std)*x + mean
            # output = self.in_layer(output)
            # mean = self.clip_mean_1(clip_feature).unsqueeze(-1).unsqueeze(-1)
            # std = self.clip_std_1(clip_feature).unsqueeze(-1).unsqueeze(-1)
            if clip_feature == None:
                clip_feature = torch.randn(x.shape[0], 512).cuda()
            output = self.in_layer(x)
            mean = self.clip_mean(clip_feature).unsqueeze(-1).unsqueeze(-1)
            std = self.clip_std(clip_feature).unsqueeze(-1).unsqueeze(-1)
            output = (1+std)*output + mean

        else:
            output = self.in_layer(x)
            if self.args.is_encode:
                output = output.view(-1, self.start_c, self.start_size, self.start_size)

        for i in range(self.args.upsample_times):
            output = self.spade_list[i](output, seg_map, clip_feature)
            output = self.upsample(output)
        output = self.spade_list[-1](output, seg_map, clip_feature)
        output = self.l_relu(output)
        output = self.act_func(self.conv_out(output))
        return output


if __name__ == "__main__":
    rank=4
    args = get_model_args()

    feature = torch.randn(1,512).to(rank)
    z = torch.randn(1, 184, 4, 4).to(rank)
    seg_map = torch.randn(1, args.seg_c, args.crop_size, args.crop_size).to(rank)
    generator = Generator(args).to(rank)
    res = generator(z, seg_map, None)

    print(res.shape)