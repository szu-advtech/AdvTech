import torch.nn
from torch.nn import functional as f
import torch.nn as nn
import torch


class Linear_unmixing_encoder(nn.Module):
    def __init__(self, config, is_rgb):
        super(Linear_unmixing_encoder, self).__init__()
        self.spectrum_band = config['chikusei']['img_band']
        self.c = config['c']
        self.encoder_no_Normalize = nn.Sequential(
            nn.Conv2d(self.spectrum_band, 8 * self.c, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(8 * self.c, 4 * self.c, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(4 * self.c, 2 * self.c, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.c, self.c, 3, 1, 1),
            nn.ReLU()
        )
        self.is_rgb = is_rgb
        self.encoder_no_Normalize_rgb = nn.Sequential(
            nn.Conv2d(3, 8 * self.c, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(8 * self.c, 4 * self.c, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(4 * self.c, 2 * self.c, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(2 * self.c, self.c, 3, 1, 1),
            nn.ReLU()
        )
        #
        # self.decoder = nn.Conv2d(self.c,self.spectrum_band, 3, 1, 1)

    def forward(self, x):  # x.shape == (batch_size,spectrum_band,height,weight)
        assert x.ndim == 4
        if self.is_rgb:
            x = self.encoder_no_Normalize_rgb(x)
            # x = self.decoder(x)
            ####################################
            x_zero = torch.zeros(x.shape)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x_zero[i, j, :, :] = x[i, j, :, :] / torch.sum(x[i, j, :, :])
            ####################################
            return x

        else:
            x = self.encoder_no_Normalize(x)
            # x = self.decoder(x)
            ####################################
            x_zero = torch.zeros(x.shape)
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    x_zero[i, j, :, :] = x[i, j, :, :] / torch.sum(x[i, j, :, :])
            ####################################
            return x


class Linear_unmixing_decoder(nn.Module):
    def __init__(self, config, is_rgb=False):
        super(Linear_unmixing_decoder, self).__init__()
        self.spectrum_band = config['chikusei']['img_band']
        self.c = config['c']
        self.decoder = nn.Conv2d(self.c, self.spectrum_band, 3, 1, 1)
        self.decoder_rgb = nn.Conv2d(self.c, 3, 3, 1, 1)
        self.is_rgb = is_rgb

    def forward(self, x):
        assert x.ndim == 4

        if self.is_rgb:
            x = self.decoder_rgb(x)
        else:
            x = self.decoder(x)

        return x


class DPCN(nn.Module):
    def __init__(self, config):
        super(DPCN, self).__init__()
        self.c = config['c']
        self.model_conv = nn.Sequential(
            nn.Conv2d(self.c, 64, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=4, dilation=4),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=3, dilation=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.ReLU(),
            nn.Conv2d(64, 4 * self.c, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.ReLU(),
            nn.modules.PixelShuffle(upscale_factor=2)
        )
        self.downsample = nn.modules.PixelUnshuffle(downscale_factor=4)
        self.conv_down = nn.Sequential(
            nn.Conv2d(16 * self.c, self.c, kernel_size=1, stride=1),
            nn.ReLU()
        )
        self.conv_up = nn.Sequential(
            nn.Conv2d(self.c, 16 * self.c, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.modules.PixelShuffle(upscale_factor=4)
        )

    def forward(self, x):
        LR = x

        # 上采样到HR空间（两次两倍上采样）
        for i in range(2):
            temp = x
            temp = f.interpolate(temp, scale_factor=2, mode='bicubic')
            x = self.model_conv(x)
            x = temp + x

        # back_projection
        rough_x = x
        x = self.downsample(x)
        x = self.conv_down(x)
        residual = LR - x
        residual = self.conv_up(residual)
        x = residual + rough_x
        return x


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)


# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out


# ResNet Module
class ResNet(nn.Module):
    def __init__(self, config, block, layers):
        super(ResNet, self).__init__()
        self.in_channels = config["c"]
        self.out_channels = self.in_channels
        self.conv = conv3x3(self.in_channels, self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, layers[0])
        self.layer2 = self.make_layer(block, layers[0], 2)
        self.layer3 = self.make_layer(block, layers[1], 2)

    def make_layer(self, block, blocks, stride=1):
        layers = [block(self.in_channels, self.out_channels, stride=1)]
        for i in range(1, blocks):
            layers.append(block(self.out_channels, self.out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        LR = x
        out = f.interpolate(x, scale_factor=4, mode='bicubic')
        out = self.conv(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        # back_projection

        rough_x = out
        out = f.interpolate(out, scale_factor=0.25, mode='bicubic')
        out = self.conv(out)
        out = self.relu(out)
        residual = LR - out
        residual = f.interpolate(residual, scale_factor=4, mode='bicubic')
        residual = self.conv(residual)
        residual = self.relu(residual)
        out = residual + rough_x
        return out
