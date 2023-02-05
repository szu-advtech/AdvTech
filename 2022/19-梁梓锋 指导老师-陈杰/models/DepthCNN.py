import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, zeros_


# 下采样的卷积层，当padding=(k-1)//2时，输出的大小就是i/stride，所以这个下采样的卷积层最终输出的长宽为原图的一半
def downsample_conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2),
        nn.ReLU(inplace=True)
    )

# 预测深度（全连接层，输出通道数为1，长宽不变）
def predict_disp(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size=3, padding=1),
        nn.Sigmoid()
    )

# 形状不变的卷积层
def conv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )

# 上采样卷积层，使用转置卷积实现，计算输出大小时可以将其当成以out为输入，in的正常卷积层，反过来解出out的形状，最后加上output_padding（长宽只加一次，也就是output_padding作用就是加n列和n行）
# 这里就是将输入图片的形状变为两倍
def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ReLU(inplace=True)
    )


# 修剪为指定的大小，直接截断多余的信息
def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]


class DispNetS(nn.Module):

    def __init__(self, alpha=10, beta=0.01):
        super(DispNetS, self).__init__()

        self.alpha = alpha
        self.beta = beta

        conv_planes = [32, 64, 128, 256, 512, 512, 512]
        # 下采样层，通道数3->32, (128*416)->(64*208)
        self.conv1 = downsample_conv(3,              conv_planes[0], kernel_size=7)
        # 下采样层，通道数32->64, (64*208)->(32*104)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size=5)
        # 下采样层，通道数64->128, (32*104)->(16*52)
        self.conv3 = downsample_conv(conv_planes[1], conv_planes[2])
        # 下采样层，通道数128->256, (16*52)->(8*26)
        self.conv4 = downsample_conv(conv_planes[2], conv_planes[3])
        # 下采样层，通道数256->512, (8*26)->(4*13)
        self.conv5 = downsample_conv(conv_planes[3], conv_planes[4])
        # 下采样层，通道数512->512, (4*13)->(2*7) 向上取整
        self.conv6 = downsample_conv(conv_planes[4], conv_planes[5])
        # 下采样层，通道数512->512, (2*7)->(1*4)  向上取整
        self.conv7 = downsample_conv(conv_planes[5], conv_planes[6])

        # 上采样层，通道数最终变为16
        upconv_planes = [512, 512, 256, 128, 64, 32, 16]
        self.upconv7 = upconv(conv_planes[6],   upconv_planes[0])
        self.upconv6 = upconv(upconv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(upconv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(upconv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(upconv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(upconv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(upconv_planes[5], upconv_planes[6])

        # iconv不影响形状，改变通道数（传入concat后的图片，然后输出的通道数为upconv_planes逆序）
        self.iconv7 = conv(upconv_planes[0] + conv_planes[5], upconv_planes[0]) # (1024->512)
        self.iconv6 = conv(upconv_planes[1] + conv_planes[4], upconv_planes[1]) # (1024->512)
        self.iconv5 = conv(upconv_planes[2] + conv_planes[3], upconv_planes[2]) # (512->256)
        self.iconv4 = conv(upconv_planes[3] + conv_planes[2], upconv_planes[3]) # (256->128) 从iconv4开始，输出会送入predict_disp4得到深度图
        self.iconv3 = conv(1 + upconv_planes[4] + conv_planes[1], upconv_planes[4]) # (129->64) 从这里开始，输入把上一层输出的深度图也concat进去，所以输入通道数+1
        self.iconv2 = conv(1 + upconv_planes[5] + conv_planes[0], upconv_planes[5]) # (65->32)
        self.iconv1 = conv(1 + upconv_planes[6], upconv_planes[6]) # (17->16)

        # predict输出通道数为1，应该就是输出深度图，不同之处在于输入的通道数不一样，“不同特征下的深度图？”
        self.predict_disp4 = predict_disp(upconv_planes[3]) #in: 128
        self.predict_disp3 = predict_disp(upconv_planes[4]) #in: 64
        self.predict_disp2 = predict_disp(upconv_planes[5]) #in: 32
        self.predict_disp1 = predict_disp(upconv_planes[6]) #in: 16

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight)
                if m.bias is not None:
                    zeros_(m.bias)

    def forward(self, x):
        out_conv1 = self.conv1(x)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6) # 1*4*512

        # 每一层上采样的过程，输出通道数和形状（通过crop_like调整）刚好是下采样的逆过程
        out_upconv7 = crop_like(self.upconv7(out_conv7), out_conv6)
        # 关键是concat，为什么要concat？而且刚好concat的两个部分是形状相同的，沿着第1维concat，也就是增大该维维数（通道数）
        concat7 = torch.cat((out_upconv7, out_conv6), 1) # （1024，2，7）
        # iconv是什么？
        out_iconv7 = self.iconv7(concat7)

        out_upconv6 = crop_like(self.upconv6(out_iconv7), out_conv5)
        concat6 = torch.cat((out_upconv6, out_conv5), 1)
        out_iconv6 = self.iconv6(concat6)

        out_upconv5 = crop_like(self.upconv5(out_iconv6), out_conv4)
        concat5 = torch.cat((out_upconv5, out_conv4), 1)
        out_iconv5 = self.iconv5(concat5)

        out_upconv4 = crop_like(self.upconv4(out_iconv5), out_conv3)
        concat4 = torch.cat((out_upconv4, out_conv3), 1)
        out_iconv4 = self.iconv4(concat4)
        disp4 = self.alpha * self.predict_disp4(out_iconv4) + self.beta

        out_upconv3 = crop_like(self.upconv3(out_iconv4), out_conv2)
        # 就像一个深度图金字塔，然后由于得到的深度图的大小不变，所以通过插值将其长宽*2，使其能够与其他输入图concat
        disp4_up = crop_like(F.interpolate(disp4, scale_factor=2, mode='bilinear', align_corners=False), out_conv2)
        concat3 = torch.cat((out_upconv3, out_conv2, disp4_up), 1)
        out_iconv3 = self.iconv3(concat3)
        disp3 = self.alpha * self.predict_disp3(out_iconv3) + self.beta

        out_upconv2 = crop_like(self.upconv2(out_iconv3), out_conv1)
        disp3_up = crop_like(F.interpolate(disp3, scale_factor=2, mode='bilinear', align_corners=False), out_conv1)
        concat2 = torch.cat((out_upconv2, out_conv1, disp3_up), 1)
        out_iconv2 = self.iconv2(concat2)
        disp2 = self.alpha * self.predict_disp2(out_iconv2) + self.beta

        out_upconv1 = crop_like(self.upconv1(out_iconv2), x)
        disp2_up = crop_like(F.interpolate(disp2, scale_factor=2, mode='bilinear', align_corners=False), x)
        concat1 = torch.cat((out_upconv1, disp2_up), 1)
        out_iconv1 = self.iconv1(concat1)
        disp1 = self.alpha * self.predict_disp1(out_iconv1) + self.beta

        if self.training:
            return disp1, disp2, disp3, disp4
        else:
            # 不是训练的模式的话，只输出disp1（也就是形状与输入相同的深度图）
            return disp1
