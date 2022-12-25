from typing import Callable, List, Optional

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from functools import partial

try:
    from torchvision.models.utils import load_state_dict_from_url # torchvision 0.4+
except ModuleNotFoundError:
    try:
        from torch.hub import load_state_dict_from_url # torch 1.x
    except ModuleNotFoundError:
        from torch.utils.model_zoo import load_url as load_state_dict_from_url # torch 0.4.1


model_urls = {
    'mobilenet_v3': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth',
}
model_urls2 = {
    'mobilenet_v3lar': 'https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth',
}


#   这个函数的目的是确保Channel个数能被8整除。
def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    # int(v + divisor / 2) // divisor * divisor：四舍五入到8
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


#   Conv+BN+Act经常会用到，组在一起
class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,         # 卷积后的BN层
                 activation_layer: Optional[Callable[..., nn.Module]] = None):  # 激活函数
        padding = (kernel_size - 1) // 2
        if norm_layer is None:          # 没有传入，就默认使用BN
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6
        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),           # 后面会用到BN层，故不使用bias
                                               norm_layer(out_planes),
                                               activation_layer(inplace=True))


#   注意力模块：SE模块
class SqueezeExcitation(nn.Module):
    # squeeze_factor: int = 4：第一个FC层节点个数是输入特征矩阵的1/4
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        # 第一个FC层节点个数，也要是8的整数倍
        squeeze_c = _make_divisible(input_c // squeeze_factor, 8)
        # 通过卷积核大小为1x1的卷积替代FC层，作用相同
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        # x有很多channel，通过output_size=(1, 1)实现每个channel变成1个数字
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        # 此处的scale就是第二个FC层输出的数据
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x        # 和原输入相乘，得到SE模块的输出


#   倒置残差块参数配置文件
class InvertedResidualConfig:
    def __init__(self,
                 input_c: int,
                 kernel: int,
                 expanded_c: int,
                 out_c: int,
                 use_se: bool,
                 activation: str,
                 stride: int,
                 width_multi: float):
        self.input_c = self.adjust_channels(input_c, width_multi)
        self.kernel = kernel
        self.expanded_c = self.adjust_channels(expanded_c, width_multi)
        self.out_c = self.adjust_channels(out_c, width_multi)
        self.use_se = use_se
        # activation == "HS"，则self.use_hs==True
        self.use_hs = activation == "HS"  # whether using h-swish activation
        self.stride = stride

    # 静态方法
    @staticmethod
    def adjust_channels(channels: int, width_multi: float):
        return _make_divisible(channels * width_multi, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        # 是否使用shortcut连接
        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers: List[nn.Module] = []    # 定义一个空列表，里面元素类型为nn.module
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.append(ConvBNActivation(cnf.input_c,
                                           cnf.expanded_c,
                                           kernel_size=1,
                                           norm_layer=norm_layer,
                                           activation_layer=activation_layer))

        # depthwise
        layers.append(ConvBNActivation(cnf.expanded_c,      # 上一层1x1输出通道数为cnf.expanded_c
                                       cnf.expanded_c,
                                       kernel_size=cnf.kernel,
                                       stride=cnf.stride,
                                       groups=cnf.expanded_c,       # DW卷积
                                       norm_layer=norm_layer,
                                       activation_layer=activation_layer))

        if cnf.use_se:      # 是否使用se模块，只需要传入个input_channel
            layers.append(SqueezeExcitation(cnf.expanded_c))

        # project       降维1x1卷积层
        layers.append(ConvBNActivation(cnf.expanded_c,
                                       cnf.out_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       # nn.Identity是一个线性激活，没进行任何处理
                                       #    内部实现：直接return input
                                       activation_layer=nn.Identity))

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class MobileNetV3(nn.Module):
    def __init__(self,
                 inverted_residual_setting: List[InvertedResidualConfig],
                 last_channel: int,         # 倒数第二层channel个数
                 num_classes: int = 1000,
                 block: Optional[Callable[..., nn.Module]] = None,
                 norm_layer: Optional[Callable[..., nn.Module]] = None):
        super(MobileNetV3, self).__init__()

        if not inverted_residual_setting:
            raise ValueError("The inverted_residual_setting should not be empty.")
        elif not (isinstance(inverted_residual_setting, List) and
                  all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])):
            raise TypeError("The inverted_residual_setting should be List[InvertedResidualConfig]")

        if block is None:
            block = InvertedResidual

        # 将norm_layer设置为BN
        #   partial()给输入函数BN指定默认参数，简化之后的函数参数量
        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []

        # building first layer
        firstconv_output_c = inverted_residual_setting[0].input_c
        layers.append(ConvBNActivation(3,
                                       firstconv_output_c,
                                       kernel_size=3,
                                       stride=2,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        # building last several layers
        lastconv_input_c = inverted_residual_setting[-1].out_c
        lastconv_output_c = 6 * lastconv_input_c            # small：96->576; Large:160->960
        # lastconv_output_c = 320
        layers.append(ConvBNActivation(lastconv_input_c,
                                       lastconv_output_c,
                                       kernel_size=1,
                                       norm_layer=norm_layer,
                                       activation_layer=nn.Hardswish))
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(lastconv_output_c, last_channel),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.2, inplace=True),
                                        nn.Linear(last_channel, num_classes))

        # initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)     # 到这后面不再需要高和宽的维度了
        x = torch.flatten(x, 1) # 故进行展平处理
        x = self.classifier(x)

        return x

    # mobilenet特征层输出
    def _forward_sal(self, x: Tensor) -> Tensor:
        res = []
        for idx, m in enumerate(self.features):
            x = m(x)
            # 只取block的第[1, 3, 6, 13，17] ！！！！！！
            # if idx in [1, 2, 4, 9, 12]:
            #     res.append(x)
            # large：：：[2, 4, 7, 13, 16]
            # small：：：[1, 2, 4, 9, 12]
            if idx in [2, 4, 7, 13, 16]:
                res.append(x)
        return res


    def forward(self, x: Tensor) -> Tensor:
        # return self._forward_impl(x)
        return self._forward_sal(x)


def mobilenet_v3_large(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:

    width_multi = 1.0       # 调整channel个数，默认1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        bneck_conf(16, 3, 16, 16, False, "RE", 1),
        bneck_conf(16, 3, 64, 24, False, "RE", 2),  # C1、24
        bneck_conf(24, 3, 72, 24, False, "RE", 1),
        bneck_conf(24, 5, 72, 40, True, "RE", 2),   # C2、40
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 5, 120, 40, True, "RE", 1),
        bneck_conf(40, 3, 240, 80, False, "HS", 2),  # C3、80
        bneck_conf(80, 3, 200, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 184, 80, False, "HS", 1),
        bneck_conf(80, 3, 480, 112, True, "HS", 1),
        bneck_conf(112, 3, 672, 112, True, "HS", 1),
        bneck_conf(112, 5, 672, 160 // reduce_divider, True, "HS", 2),  # C4、160
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
        bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider, 160 // reduce_divider, True, "HS", 1),
    ]
    last_channel = adjust_channels(1280 // reduce_divider)  # C5    # 倒数第二个全连接层节点个数

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


def mobilenet_v3_small(num_classes: int = 1000,
                       reduced_tail: bool = False) -> MobileNetV3:
    width_multi = 1.0
    bneck_conf = partial(InvertedResidualConfig, width_multi=width_multi)
    adjust_channels = partial(InvertedResidualConfig.adjust_channels, width_multi=width_multi)

    reduce_divider = 2 if reduced_tail else 1

    inverted_residual_setting = [
        # input_c, kernel, expanded_c, out_c, use_se, activation, stride
        # C = [1,2,4,9,12]
        bneck_conf(16, 3, 16, 16, True, "RE", 2),  # C1
        bneck_conf(16, 3, 72, 24, False, "RE", 2),  # C2
        bneck_conf(24, 3, 88, 24, False, "RE", 1),
        bneck_conf(24, 5, 96, 40, True, "HS", 2),  # C3
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 240, 40, True, "HS", 1),
        bneck_conf(40, 5, 120, 48, True, "HS", 1),
        bneck_conf(48, 5, 144, 48, True, "HS", 1),
        bneck_conf(48, 5, 288, 96 // reduce_divider, True, "HS", 2),  # C4
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1),
        bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider, 96 // reduce_divider, True, "HS", 1)
    ]
    last_channel = adjust_channels(1024 // reduce_divider)  # C5

    return MobileNetV3(inverted_residual_setting=inverted_residual_setting,
                       last_channel=last_channel,
                       num_classes=num_classes)


# 调用small
def mobilenet_v3(pretrained=True, progress=True, **kwargs):
    model = mobilenet_v3_small(**kwargs)
    if pretrained:
        # 下载mobilenet_v3_samll的预训练模型
        state_dict = load_state_dict_from_url(model_urls['mobilenet_v3'],
                                              progress=progress)
        print("loading imagenet pretrained mobilenetv3_SMALL")
        model.load_state_dict(state_dict, strict=False)
        print("loaded imagenet pretrained mobilenetv3_SMALL")
    return model


# 调用large
def mobilenet_v3lar(pretrained=True, progress=True, **kwargs):
    model = mobilenet_v3_large(**kwargs)
    if pretrained:
        # 下载mobilenet_v3_large的预训练模型
        state_dict = load_state_dict_from_url(model_urls2['mobilenet_v3lar'],
                                              progress=progress)
        print("loading imagenet pretrained mobilenet_v3LARGE")
        model.load_state_dict(state_dict, strict=False)
        print("loaded imagenet pretrained mobilenet_v3LARGE")
    return model

# if __name__ == "__main__":
#     model = mobilenet_v3()
#     print(model)
#
#     input = torch.randn(1, 3, 224, 224)
#     out = model(input)
#     print(out.shape)

    # from torchsummaryX import summary
    # summary(model, torch.randn(1,3,224,224))
