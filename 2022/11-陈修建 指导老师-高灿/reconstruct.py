
from torch import nn
# 3×3的卷积
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)
# 1×1的卷积
def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 重构学生网络
class ReconstructiveStudent(nn.Module):
    def __init__(self):
        super(ReconstructiveStudent, self).__init__()
        in_channels = 512
        self.inplanes = in_channels
        channels_1024 = 2*in_channels
        channels_512 = in_channels
        channels_256 = in_channels // 2
        self.norm_layer = nn.BatchNorm2d
        self.relu = nn.ReLU(inplace=True)
        self.conv_1 = conv3x3(in_channels, channels_1024)
        self.bn1 = self.norm_layer(channels_1024)
        self.block1 = BasicBlock(channels_1024, channels_1024)
        self.deconv_1 = nn.ConvTranspose2d(in_channels=channels_1024, out_channels=channels_1024, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = self.norm_layer(channels_1024)
        self.layer1 = self._make_layer(BasicBlock, channels_1024, 2)
        # 添加注意力机制
        self.atten1 = MyAttention(channels_1024)
        self.deconv_2 = nn.ConvTranspose2d(in_channels=channels_1024, out_channels=channels_512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = self.norm_layer(channels_512)
        self.layer2 = self._make_layer(BasicBlock, channels_512, 2, True)
        self.atten2 = MyAttention(channels_512)
        self.deconv_3 = nn.ConvTranspose2d(in_channels=channels_512, out_channels=channels_256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = self.norm_layer(channels_256)
        self.layer3 = self._make_layer(BasicBlock, channels_256, 2, True)

    def forward(self, x, att1, att2):
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.deconv_1(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)
        a1 = self.atten1(att1)
        x = x*a1
        x = self.deconv_2(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.layer2(x)
        a2 = self.atten2(att2)
        x = x*a2+x
        x = self.deconv_3(x)
        x = self.bn4(x)
        x = self.layer3(x)
        return x
    def _make_layer(self, block, planes: int, blocks: int, need_downsample = False,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self.norm_layer
        downsample = None
        if dilate:
            self.dilation *= stride
            stride = 1
        if need_downsample:
            downsample = nn.Sequential(
                conv1x1(planes, planes, stride),
                norm_layer(planes),
            )

        layers = []
        layers.append(block(planes, planes, stride, downsample, 1, norm_layer=norm_layer))
        for _ in range(1, blocks):
            layers.append(block(planes, planes, norm_layer=norm_layer))

        return nn.Sequential(*layers)
# 注意力机制
class MyAttention(nn.Module):
    def __init__(self, inplanes):
        super(MyAttention, self).__init__()
        self.conv_1 = conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv_2 = conv1x1(inplanes, 1)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.sigmod(x)
        return x
# 基本块
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1,
        dilation: int = 1,
        norm_layer = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


