import jittor as jt 
from jittor import nn
import warnings

from jdet.utils.registry import NECKS
from jdet.models.utils.modules import ConvModule
from jdet.models.utils.weight_init import xavier_init
def func_2(a,b):
    #求两个数的最大公因数
    t=a%b
    #当余数不为零的时候继续循环，当为0的时候输出结果
    while t!=0:
        a=b;
        b=t;
        t=a%b
    return b
def add_conv(in_ch, out_ch, ksize, stride, leaky=True):
    """
    Add a conv2d / batchnorm / leaky ReLU block.
    Args:
        in_ch (int): number of input channels of the convolution layer.
        out_ch (int): number of output channels of the convolution layer.
        ksize (int): kernel size of the convolution layer.
        stride (int): stride of the convolution layer.
    Returns:
        stage (Sequential) : Sequential layers composing a convolution block.
    """
    stage = nn.Sequential()
    pad = (ksize - 1) // 2
    stage.add_module('conv', nn.Conv2d(in_channels=in_ch,
                                       out_channels=out_ch, kernel_size=ksize, stride=stride,
                                       padding=pad, bias=False,groups=func_2(in_ch,out_ch)))
    stage.add_module('batch_norm', nn.BatchNorm2d(out_ch))
    if leaky:
        stage.add_module('leaky', nn.LeakyReLU(0.1))
    else:
        stage.add_module('relu6', nn.ReLU6(inplace=True))
    return stage
class ASFF(nn.Module):
    def __init__(self, level=0,indim=[512,256,256,256],out_dim=[1024,512,256,256], rfb=False, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = indim
        self.out_dim = out_dim
        self.inter_dim = self.dim[self.level]
        if level==0:
        	# self.stride_level_1用来调整level-1出来的特征图，通道调整为和level-1出来的特征图一样大小
        	# (3,3)的卷积核stride=2，通道调整为512，分辨率降采样2倍，扩大感受野
            self.stride_level_1 = add_conv(self.dim[1], self.inter_dim, 3, 2)
            # self.stride_level_2用来调整level-2出来的特征图，保持和level-1出来的特征图一样大小
            # (3,3)的卷积核stride=2，通道调整为512，分辨率降采样2倍，扩大感受野
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)

            self.stride_level_3 = add_conv(self.dim[3],self.inter_dim,3,8)
            # 调整融合后的权重的通道并扩大感受野
            self.expand = add_conv(self.inter_dim, self.out_dim[0], 3, 1)
        elif level==1:
        	# 与level==0同理，都是为了将其余的level出来的特征图调整到当前leval出来的特征图一样的大小
        	# 否则就没法加权融合了
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            self.stride_level_2 = add_conv(self.dim[2], self.inter_dim, 3, 2)

            self.stride_level_3 = add_conv(self.dim[3],self.inter_dim, 3, 4)
            self.expand = add_conv(self.inter_dim, self.out_dim[1], 3, 1)
        elif level==2:
        	# 与上面同理
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            self.compress_level_1 = add_conv(self.dim[1], self.inter_dim, 1, 1)
            self.stride_level_3 = add_conv(self.dim[3],self.inter_dim, 3, 2)
            self.expand = add_conv(self.inter_dim, self.out_dim[2], 3, 1)
        elif level==3:
            self.compress_level_0 = add_conv(self.dim[0], self.inter_dim, 1, 1)
            self.compress_level_1 = add_conv(self.dim[1], self.inter_dim, 1, 1)
            self.compress_level_2 = add_conv(self.dim[2], self.inter_dim, 1, 1)
            self.expand = add_conv(self.inter_dim, self.out_dim[3], 3, 1)

		
        compress_c = 8 if rfb else 16  #when adding rfb, we use half number of channels to save memory
		# 用来得到权重的三个卷积
        self.weight_level_0 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = add_conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = add_conv(self.inter_dim, compress_c, 1, 1)
		# 用于调整chanel拼接后的三个权重的通道
        self.weight_levels = nn.Conv2d(compress_c*4, 4, kernel_size=1, stride=1, padding=0)
        self.vis= vis


    def execute(self, x_level_0, x_level_1, x_level_2,x_level_3):
        if self.level==0:
        	# 以level==0为例后面类似
        	# 在level==0时，level-0出来的特征图保持不变
        	# 调整其他两个特征图以使得level-1、level-2的特征图的chanel、width、height与level-0相同
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)

            level_2_downsampled_inter =nn.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)

            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level==1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =nn.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized =x_level_1
            level_2_resized =self.stride_level_2(x_level_2)
            level_3_downsampled_inter =nn.max_pool2d(x_level_3, 3, stride=2, padding=1)
            level_3_resized = self.stride_level_3(x_level_3)

        elif self.level==2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =nn.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized =nn.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized =x_level_2
            level_3_resized = self.stride_level_3(x_level_3)
        elif self.level == 3:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized =nn.interpolate(level_0_compressed, scale_factor=8, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized =nn.interpolate(level_1_compressed, scale_factor=4, mode='nearest')

            level_2_compressed = self.compress_level_2(x_level_2)
            level_2_resized =nn.interpolate(level_2_compressed, scale_factor=2, mode='nearest')
            level_3_resized =x_level_3
            
		# 将N*C*H*W的level-0特征图卷积得到权重，权重level_0_weight_v:N*256*H*W
        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        level_3_weight_v = self.weight_level_3(level_3_resized)
        # 将各个权重矩阵按照通道concat，levels_weight_v：N*3C*H*W
        levels_weight_v = jt.concat((level_0_weight_v, level_1_weight_v, level_2_weight_v,level_3_weight_v),1)
        # 将concat后的矩阵调整3通道，每个通道对应着不同的level_0_resized，level_1_resized，level_2_resized的权重
        levels_weight = self.weight_levels(levels_weight_v)
        # 在通道维度，对权重做归一化，也就是对于对于三通道tmp：tmp[0][0][0]+tmp[1][0][0]+tmp[2][0][0]=1
        levels_weight = nn.softmax(levels_weight, dim=1)
		# 将levels_weight各个通道分别乘level_0_resized level_1_resized level_2_resized
		# level_0_resized是N*C*H*W而levels_weight[:,0:1,:,:]是N*1*H*W
		# 点乘用到了广播机制
        fused_out_reduced = level_0_resized * levels_weight[:,0:1,:,:]+\
                            level_1_resized * levels_weight[:,1:2,:,:]+\
                            level_2_resized * levels_weight[:,2:3,:,:]+\
                            level_3_resized * levels_weight[:,3:,:,:]
		# 3*3调整一下通道特征图分辨率不变
        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out
class ASFF_Block4(nn.Module):
    def __init__(self,indim=[256,256,256,256],out_dim=[256,256,256,256]):
        super(ASFF_Block4, self).__init__()
        self.assf0 = ASFF(level=0,indim=indim,out_dim=out_dim)
        self.assf1 = ASFF(level=1,indim=indim,out_dim=out_dim)
        self.assf2 = ASFF(level=2,indim=indim,out_dim=out_dim)
        self.assf3 = ASFF(level=3,indim=indim,out_dim=out_dim)
    def execute(self,x):
        xin0 = x[-1]
        xin1 = x[-2]
        xin2 = x[-3]
        xin3 = x[-4]
        xout0 = self.assf0(xin0,xin1,xin2,xin3)
        xout1 = self.assf1(xin0,xin1,xin2,xin3)
        xout2 = self.assf2(xin0,xin1,xin2,xin3)
        xout3 = self.assf3(xin0,xin1,xin2,xin3)
        # print(f"kokokokokokoooooooooooooooooooooooooooooooookokoko")
        return tuple([xout3,xout2,xout1,xout0])
@NECKS.register_module()
class FPN_ASSF_Block4(nn.Module):
    r"""Feature Pyramid Network.
    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.
    Args:
        in_channels (List[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale)
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, its actual mode is specified by `extra_convs_on_inputs`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed
            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral':  Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        extra_convs_on_inputs (bool, deprecated): Whether to apply extra convs
            on the original feature from the backbone. If True,
            it is equivalent to `add_extra_convs='on_input'`. If False, it is
            equivalent to set `add_extra_convs='on_output'`. Default to True.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (str): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: `dict(mode='nearest')`
        init_cfg (dict or list[dict], optional): Initialization config dict.
    Example:
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [jt.randn(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = ([1, 11, 340, 340])
        outputs[1].shape = ([1, 11, 170, 170])
        outputs[2].shape = ([1, 11, 84, 84])
        outputs[3].shape = ([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform'),
                 upsample_div_factor=1):
        super(FPN_ASSF_Block4, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.upsample_cfg = upsample_cfg.copy()
        self.upsample_div_factor = upsample_div_factor
        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg)
                self.fpn_convs.append(extra_fpn_conv)
        self.assf_block4 = ASFF_Block4(indim=[out_channels]*4,out_dim=[out_channels]*4)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution='uniform')

    def execute(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] += nn.interpolate(laterals[i],
                                                 **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] += nn.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)
            laterals[i - 1] /= self.upsample_div_factor
        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(nn.pool(outs[-1], 1, stride=2,op="maximum"))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](nn.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        t = list(self.assf_block4(tuple(outs[0:-1])))
        t.append(outs[-1])
        # print(len(t))
        # print(f"t[0].shape:{t[0].shape},t[1].shape:{t[1].shape},t[2].shape:{t[2].shape}")
        return tuple(t)
        # return self.assf_block4(tuple(outs))