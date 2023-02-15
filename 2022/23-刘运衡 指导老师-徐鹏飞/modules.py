import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from utils import per_pix_dot, unpack_svbrdf, uv2normal, tensor_show

# 自定义STconv层
class STconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(STconv, self).__init__()
        # 二维卷积
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        # 激活函数LeakyReLU
        # self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.activation = activation

    def forward(self, x):
        # 卷积
        x = self.conv(x)
        # 激活
        x = self.activation(x)
        return x

# 自定义inception层


class Inception(nn.Module):
    def __init__(self, in_channels, out_channels, padding, stride=1, dilation=1):
        super(Inception, self).__init__()
        self.out_channels = out_channels//2
        # 定义第一个3*3标准卷积层
        self.conv1 = STconv(in_channels=in_channels, out_channels=self.out_channels,
                            kernel_size=3, stride=stride, padding=padding, dilation=dilation)
        # 定义两个连续的3*3标准卷积层
        self.conv2 = nn.Sequential(
            STconv(in_channels=in_channels, out_channels=self.out_channels,
                   kernel_size=3, stride=1, padding=padding, dilation=dilation),
            STconv(in_channels=self.out_channels, out_channels=self.out_channels,
                   kernel_size=3, stride=stride, padding=padding, dilation=dilation)
        )

    def forward(self, x):
        # 对两个3*3卷积层进行拼接
        # print("@@@@@@@@@@@")
        # print(x.shape)
        # print(self.conv1(x).shape)
        # print(self.conv2(x).shape)
        # print("###############")
        x = torch.cat([self.conv1(x), self.conv2(x)], dim=1)
        return x

# 自定义HA卷积层


class HAconv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation=1):
        super(HAconv, self).__init__()
        # HA卷积块的最上层,先进行卷积，再进行sigmoid激活
        self.conv_top = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.Sigmoid()
        )
        # HA卷积块的中间层，先进行instance normalization，再进行卷积，再进行LeakyReLU激活
        self.conv_mid = nn.Sequential(
            nn.InstanceNorm2d(out_channels),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                      kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # HA卷积块的最下层，也就是Inception层
        self.conv_bottom = Inception(
            in_channels=in_channels, out_channels=out_channels, stride=stride, padding=padding, dilation=dilation)

    def forward(self, x):
        # 将HA卷积最上层与中层先进行元素乘法
        y = self.conv_top(x) * self.conv_mid(x)
        # 将得到的结果y与经过最下层的卷积层进行元素加法
        y = y + self.conv_bottom(x)
        #  LeakyReLU（在论文5.1中提到每个块后面都跟随了一个LeakyReLU）
        y = nn.LeakyReLU(0.2, inplace=True)(y)
        return y

# 自定义MLP层


class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()
        # 全连接层
        self.fc = nn.Linear(in_features=in_channels, out_features=out_channels)

    def forward(self, x):
        # 全连接层
        x = self.fc(x)
        # # 激活函数LeakyReLU(暂时未提到此处有激活函数)
        # x = nn.LeakyReLU(0.2, inplace=True)(x)
        return x
# 自定义HABranch层


class HABranch(nn.Module):
    def __init__(self):
        super(HABranch, self).__init__()
        # HA卷积层1、2、3、4，输出通道分别为64,64，64,128
        self.HAconv1 = nn.Sequential(
            HAconv(in_channels=64, out_channels=64,
                   kernel_size=3, stride=1, padding=1),
            HAconv(in_channels=64, out_channels=64,
                   kernel_size=3, stride=2, padding=1),
            HAconv(in_channels=64, out_channels=64,
                   kernel_size=3, stride=1, padding=1),
            HAconv(in_channels=64, out_channels=128,
                   kernel_size=3, stride=2, padding=1)
        )
        # HA卷积层5、6，输出通道分别为128,128
        self.HAconv2 = nn.Sequential(
            HAconv(in_channels=128, out_channels=128,
                   kernel_size=3, stride=1, padding=1),
            HAconv(in_channels=128, out_channels=128,
                   kernel_size=3, stride=2, padding=1)
        )
        # HA卷积层7，输出通道为128，步长为1，空洞卷积系数为2
        self.HAconv3 = HAconv(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=2, dilation=2)
        # ST卷积层8，双线性插值，输出通道为128，步长为1
        self.HAconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            STconv(in_channels=128, out_channels=128,
                   kernel_size=3, stride=1, padding=1)
        )
        # ST卷积层9，双线性插值，输出通道为64，步长为1
        self.HAconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            STconv(in_channels=128, out_channels=64,
                   kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # HA卷积块1
        x = self.HAconv1(x)
        # 复制x，并进行HA卷积块2
        y = x.clone()
        x = self.HAconv2(x)
        # 复制x，并进行HA卷积块3
        z = x.clone()
        x = self.HAconv3(x)
        # skip connection
        x = x + z
        # HA卷积块4
        x = self.HAconv4(x)
        # skip connection
        x = x + y
        # HA卷积块5
        x = self.HAconv5(x)
        return x

# 自定义STBranch层


class STBranch(nn.Module):
    def __init__(self):
        super(STBranch, self).__init__()
        # ST卷积，通道数为64,层1、2、3、4
        self.STconv1 = nn.Sequential(
            STconv(in_channels=64, out_channels=64,
                   kernel_size=3, stride=1, padding=1),
            STconv(in_channels=64, out_channels=64,
                   kernel_size=3, stride=2, padding=1),
            STconv(in_channels=64, out_channels=64,
                   kernel_size=3, stride=1, padding=1),
            STconv(in_channels=64, out_channels=128,
                   kernel_size=3, stride=2, padding=1),
        )
        # ST卷积，通道数为128,层5、6
        self.STconv2 = nn.Sequential(
            STconv(in_channels=128, out_channels=128,
                   kernel_size=3, stride=1, padding=1),
            STconv(in_channels=128, out_channels=128,
                   kernel_size=3, stride=2, padding=1)
        )
        # ST卷积，通道数为128,层7，步长为1,空洞卷积因子为2
        self.STconv3 = STconv(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=2, dilation=2)
        # ST卷积，通道数为128，双线性插值，步长为1
        self.STconv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            STconv(in_channels=128, out_channels=128,
                   kernel_size=3, stride=1, padding=1)
        )
        # ST卷积，通道数为64，双线性插值，步长为1
        self.STconv5 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            STconv(in_channels=128, out_channels=64,
                   kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        # ST卷积，块1
        x = self.STconv1(x)
        # 复制x，准备skip connection
        x_copy = x.clone()
        # ST卷积，块2
        x = self.STconv2(x)
        # 复制x，准备skip connection
        x_copy2 = x.clone()
        # ST卷积，块3
        x = self.STconv3(x)
        # skip connection
        x = x + x_copy2
        # ST卷积，块4
        x = self.STconv4(x)
        # skip connection
        x = x + x_copy
        # ST卷积，块5
        x = self.STconv5(x)
        return x
# 自定义AFS层


class AFS(nn.Module):
    def __init__(self):
        super(AFS, self).__init__()
        # 全局均值池化
        self.global_avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # MLP层，输入通道数为128，输出通道数为128,单个隐藏层16个节点，激活函数为Sigmoid
        self.MLP = nn.Sequential(
            # 卷积层1，输入通道数为128，输出通道数为16，卷积核大小为1*1，步长为1，padding为0
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=1, stride=1, padding=0),
            # 卷积层2，输入通道数为16，输出通道数为128，卷积核大小为1*1，步长为1，padding为0
            nn.Conv2d(in_channels=128, out_channels=128,
                      kernel_size=1, stride=1, padding=0),
            # 激活函数
            nn.Sigmoid()
        )

    def forward(self, x):
        # 全局均值池化
        y = self.global_avgpool(x)
        # MLP层
        y = self.MLP(y)
        # 将x与y相乘
        x = y * x
        return x

# 自定义FUBranch层


class FUBranch(nn.Module):
    def __init__(self, out_channels=3):
        super(FUBranch, self).__init__()
        # AFS层
        self.AFS = AFS()
        # ST卷积，通道数为128
        self.STconv1 = STconv(in_channels=128, out_channels=128,
                              kernel_size=3, stride=1, padding=1)
        # ST卷积，双线性插值，步长为1，通道为64
        self.STconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            STconv(in_channels=128, out_channels=64,
                   kernel_size=3, stride=1, padding=1)
        )
        # ST卷积,通道为64
        self.STconv3 = STconv(in_channels=64, out_channels=64,
                              kernel_size=3, stride=1, padding=1)
        # ST卷积，通道数为3
        # self.STconv4 = STconv(in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1, activation=nn.Sigmoid())
        self.STconv4 = STconv(
            in_channels=64, out_channels=out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # AFS层
        x = self.AFS(x)
        # ST卷积
        x = self.STconv1(x)
        # ST卷积
        x = self.STconv2(x)
        # ST卷积
        x = self.STconv3(x)
        # ST卷积
        x = self.STconv4(x)

        x = torch.clip(x, 0.0, 1.0)
        # print(x.shape)
        return x


class Render(nn.Module):
    def __init__(self, less_channel=False):
        super(Render, self).__init__()
        self.less_channel = less_channel

    @staticmethod
    def render_diffuse_Substance(diffuse, specular):
        return diffuse * (1.0 - specular) / np.pi

    @staticmethod
    def render_D_GGX_Substance(roughness, NdotH):
        alpha = torch.square(roughness)
        tmp = (torch.square(NdotH) * (torch.square(alpha) - 1.0) + 1.0)
        underD = 1/torch.maximum(torch.full_like(tmp, 0.001), tmp)
        return (torch.square(alpha * underD)/np.pi)

    @staticmethod
    def render_F_GGX_Substance(specular, VdotH):
        sphg = torch.pow(2.0, ((-5.55473 * VdotH) - 6.98316) * VdotH)
        return specular + (1.0 - specular) * sphg

    @staticmethod
    def render_G_GGX_Substance(roughness, NdotL, NdotV):
        return Render.G1_Substance(NdotL, torch.square(roughness)/2) * Render.G1_Substance(NdotV, torch.square(roughness)/2)

    @staticmethod
    def G1_Substance(NdotW, k):
        tmp = (NdotW * (1.0 - k) + k)
        return 1.0/torch.maximum(tmp, torch.full_like(tmp, 0.001))

    @staticmethod
    def squeezeValues(tensor, min, max):
        return torch.clip(tensor, min, max)

    @staticmethod
    def deprocess(image):
        print(image.max(), image.min())
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

    def forward(self, svbrdf, wi, wo, includeDiffuse=True):
        wiNorm = F.normalize(wi, dim=1).to('cuda')
        woNorm = F.normalize(wo, dim=1).to('cuda')
        h = F.normalize((wiNorm + woNorm) / 2.0, dim=1).to('cuda')

        normals, diffuse, roughness, specular = unpack_svbrdf(
            svbrdf, less_channel=self.less_channel)
        specular = self.squeezeValues(specular, 0.0, 1.0)
        roughness = self.squeezeValues(roughness, 0.0, 1.0)
        roughness = torch.maximum(roughness, torch.full_like(roughness, 0.001))
        diffuse = self.squeezeValues(diffuse, 0.0, 1.0)
        if self.less_channel:
            # 把角度转回法线
            normals = uv2normal((normals-0.5)*torch.pi)
            roughness = torch.tile(roughness, dims=[1, 3, 1, 1])

        NdotH = per_pix_dot(normals, h, dim=1)
        NdotL = per_pix_dot(normals, wiNorm, dim=1)
        NdotV = per_pix_dot(normals, woNorm, dim=1)
        VdotH = per_pix_dot(woNorm, h, dim=1)

        diffuse_rendered = self.render_diffuse_Substance(diffuse, specular)
        D_rendered = self.render_D_GGX_Substance(roughness, F.relu(NdotH))
        G_rendered = self.render_G_GGX_Substance(
            roughness, F.relu(NdotL), F.relu(NdotV))
        F_rendered = self.render_F_GGX_Substance(specular, F.relu(VdotH))

        specular_rendered = F_rendered * (G_rendered * D_rendered * 0.25)
        result = specular_rendered

        if includeDiffuse:
            result = result + diffuse_rendered

        lampIntensity = 1.0

        lampFactor = lampIntensity * torch.pi

        result = result * lampFactor

        result = result * F.relu(NdotL) / torch.unsqueeze(torch.maximum(
            wiNorm[:, 2, :, :], torch.full_like(wiNorm[:, 2, :, :], 0.001)), axis=0)

        return [result, D_rendered, G_rendered, F_rendered, diffuse_rendered, diffuse]


# 定义pytorch模型
class HAModel(nn.Module):
    def __init__(self, config=None, less_channel=False):
        super(HAModel, self).__init__()
        # 定义卷积层1，通道为64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        # modules的STRranch块
        self.STBranch = STBranch()
        # modules的HABranch块
        self.HABranch = HABranch()
        # 四个modules的FUBranch块
        self.FUBranch1 = FUBranch(out_channels=2 if less_channel else 3)
        self.FUBranch2 = FUBranch()
        self.FUBranch3 = FUBranch(out_channels=1 if less_channel else 3)
        self.FUBranch4 = FUBranch()

    # 定义前向传播
    def forward(self, x):
        # 卷积层1
        x = self.conv1(x)
        # 将x分别进行HABranch和STBranch，之后进行concate
        x = torch.cat((self.HABranch(x), self.STBranch(x)), 1)
        # 经过4次FUBranch，产生4个输出
        x1 = self.FUBranch1(x)
        x2 = self.FUBranch2(x)
        x3 = self.FUBranch3(x)
        x4 = self.FUBranch4(x)
        # 返回结果
        return x1, x2, x3, x4

    # 定义计算特征层的维度
    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
