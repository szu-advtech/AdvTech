""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        self.bc_up1 = Up(1024, 512 // factor, bilinear)
        self.bc_up2 = Up(512, 256 // factor, bilinear)
        self.bc_up3 = Up(256, 128 // factor, bilinear)
        self.bc_up4 = Up(128, 64, bilinear)
        self.bc_outc = OutConv(64, 3)

        self.r_up1 = Up(1024, 512 // factor, bilinear)
        self.r_up2 = Up(512, 256 // factor, bilinear)
        self.r_up3 = Up(256, 128 // factor, bilinear)
        self.r_up4 = Up(128, 64, bilinear)
        self.r_outc = OutConv(64, 1)

        self.m_up1 = Up(1024, 512 // factor, bilinear)
        self.m_up2 = Up(512, 256 // factor, bilinear)
        self.m_up3 = Up(256, 128 // factor, bilinear)
        self.m_up4 = Up(128, 64, bilinear)
        self.m_outc = OutConv(64, 1)

        self.n_up1 = Up(1024, 512 // factor, bilinear)
        self.n_up2 = Up(512, 256 // factor, bilinear)
        self.n_up3 = Up(256, 128 // factor, bilinear)
        self.n_up4 = Up(128, 64, bilinear)
        self.n_outc = OutConv(64, 3)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        bc_x = self.bc_up1(x5, x4)
        bc_x = self.bc_up2(bc_x, x3)
        bc_x = self.bc_up3(bc_x, x2)
        bc_x = self.bc_up4(bc_x, x1)
        bc_logits = self.bc_outc(bc_x)

        r_x = self.r_up1(x5, x4)
        r_x = self.r_up2(r_x, x3)
        r_x = self.r_up3(r_x, x2)
        r_x = self.r_up4(r_x, x1)
        r_logits = self.r_outc(r_x)

        m_x = self.m_up1(x5, x4)
        m_x = self.m_up2(m_x, x3)
        m_x = self.m_up3(m_x, x2)
        m_x = self.m_up4(m_x, x1)
        m_logits = self.m_outc(m_x)

        n_x = self.n_up1(x5, x4)
        n_x = self.n_up2(n_x, x3)
        n_x = self.n_up3(n_x, x2)
        n_x = self.n_up4(n_x, x1)
        n_logits = self.n_outc(n_x)

        out = torch.cat([bc_logits, r_logits, m_logits, n_logits], dim=1)
        # print(out.size())


        return out
