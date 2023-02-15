import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# The basic block in ResNet-34
class basic_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, r_block=None):
        super(basic_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.right_block = r_block
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv(x)
        if self.right_block is not None:
            identity = self.right_block(x)

        out += identity
        out = self.relu(out)
        return out


# make layers in ResNet-34 layer2-layer5
def make_layers(in_channels, out_channels, num_block, stride):
    r_block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
        nn.BatchNorm2d(out_channels)
    )

    layer = [basic_block(in_channels, out_channels, stride=stride, r_block=r_block)]

    for i in range(1, num_block):
        layer.append(basic_block(out_channels, out_channels, stride=1))
    return nn.Sequential(*layer)


# Q: ResNet34 need leakyReLU? (un modify)
class res34_unet(nn.Module):
    def __init__(self):
        super(res34_unet, self).__init__()
        # pre-process
        # input: a render image(3 channels) and a mask(1 channels) = 4 channels
        # prepare module, according to ABO_supplement, we need to use BatchNorm and LeakyReLU
        self.pre = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(64),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(64)
        )
        # encoder - uses a ResNet-34(conv1- conv5) backbone. We use ReLU in ResNet-34 instead of LeakyReLU provided
        # by themselves.
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv2_max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = make_layers(64, 64, 3, 1)
        self.conv3 = make_layers(64, 128, 4, 2)
        self.conv4 = make_layers(128, 256, 6, 2)
        self.conv5 = make_layers(256, 512, 3, 2)

        # The decoders follow the request in paper:
        # "We use BatchNorm and leaky ReLU"
        # After we complete ConvTranspose, we would use BatchNorm and LeakyReLU as follows.
        # base_color decoder
        self.db_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.db_conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.db_conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.db_conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.db_conv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        # post process
        self.bpost_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.bpost_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True)
        )

        # metallic
        self.dm_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.dm_conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.dm_conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.dm_conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.dm_conv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        # post process
        self.mpost_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.mpost_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True)
        )

        # roughness
        self.dr_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.dr_conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.dr_conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.dr_conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.dr_conv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        # post process
        self.rpost_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.rpost_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True)
        )

        # normal
        self.dn_conv1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )
        self.dn_conv2 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(512, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True)
        )
        self.dn_conv3 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(256, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.dn_conv4 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.dn_conv5 = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        # post process
        self.npost_1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True)
        )
        self.npost_2 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=1, padding=1, output_padding=0),
            nn.BatchNorm2d(3),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        x = self.pre(x)
        x1 = self.conv1(x)
        x2 = self.conv2_max_pool(x1)
        x2 = self.conv2(x2)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # base color
        out = self.db_conv1(x5)
        # crop x4
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x4 = crop(x4)
        out = torch.cat([x4, out], dim=1)
        out = self.db_conv2(out)
        # crop x3
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x3 = crop(x3)
        out = torch.cat([x3, out], dim=1)
        out = self.db_conv3(out)
        # crop x2
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x2 = crop(x2)
        out = torch.cat([x2, out], dim=1)
        out = self.db_conv4(out)
        # crop x1
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x1 = crop(x1)
        out = torch.cat([x1, out], dim=1)
        out = self.db_conv5(out)
        # crop x
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x = crop(x)
        out = torch.cat([x, out], dim=1)
        out = self.bpost_1(out)
        bout = self.bpost_2(out)

        # metallic
        out = self.dm_conv1(x5)
        # crop x4
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x4 = crop(x4)
        out = torch.cat([x4, out], dim=1)
        out = self.dm_conv2(out)
        # crop x3
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x3 = crop(x3)
        out = torch.cat([x3, out], dim=1)
        out = self.dm_conv3(out)
        # crop x2
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x2 = crop(x2)
        out = torch.cat([x2, out], dim=1)
        out = self.dm_conv4(out)
        # crop x1
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x1 = crop(x1)
        out = torch.cat([x1, out], dim=1)
        out = self.dm_conv5(out)
        # crop x
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x = crop(x)
        out = torch.cat([x, out], dim=1)
        out = self.mpost_1(out)
        mout = self.mpost_2(out)

        # roughness
        out = self.dr_conv1(x5)
        # crop x4
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x4 = crop(x4)
        out = torch.cat([x4, out], dim=1)
        out = self.dr_conv2(out)
        # crop x3
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x3 = crop(x3)
        out = torch.cat([x3, out], dim=1)
        out = self.dr_conv3(out)
        # crop x2
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x2 = crop(x2)
        out = torch.cat([x2, out], dim=1)
        out = self.dr_conv4(out)
        # crop x1
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x1 = crop(x1)
        out = torch.cat([x1, out], dim=1)
        out = self.dr_conv5(out)
        # crop x
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x = crop(x)
        out = torch.cat([x, out], dim=1)
        out = self.rpost_1(out)
        rout = self.rpost_2(out)

        # normal
        out = self.dn_conv1(x5)
        # crop x4
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x4 = crop(x4)
        out = torch.cat([x4, out], dim=1)
        out = self.dn_conv2(out)
        # crop x3
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x3 = crop(x3)
        out = torch.cat([x3, out], dim=1)
        out = self.dn_conv3(out)
        # crop x2
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x2 = crop(x2)
        out = torch.cat([x2, out], dim=1)
        out = self.dn_conv4(out)
        # crop x1
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x1 = crop(x1)
        out = torch.cat([x1, out], dim=1)
        out = self.dn_conv5(out)
        # crop x
        crop = torchvision.transforms.CenterCrop((out.size()[3], out.size()[2]))
        x = crop(x)
        out = torch.cat([x, out], dim=1)
        out = self.npost_1(out)
        nout = self.npost_2(out)

        # cat all output(base_color, metallic, roughness, normal) -> torch.size([batch_size, 8, 256, 256])
        out = torch.cat([bout, rout, mout, nout], dim=1)

        return out
