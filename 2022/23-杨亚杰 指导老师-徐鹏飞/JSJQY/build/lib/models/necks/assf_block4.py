
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
@NECKS.register_module()
class ASFF_Block4(nn.Module):
    def __init__(self,indim=[512,256,256,256],out_dim=[1024,512,256,256]):
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
        return tuple([xout3,xout2,xout1,xout0])