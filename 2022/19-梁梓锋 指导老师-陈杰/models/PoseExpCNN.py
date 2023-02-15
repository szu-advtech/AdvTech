import torch
import torch.nn as nn
from torch import sigmoid
from torch.nn.init import xavier_uniform_, zeros_

# 卷积层，下采样层，长宽减半
def conv(in_planes, out_planes, kernel_size=3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=(kernel_size-1)//2, stride=2),
        nn.ReLU(inplace=True)
    )

# 上采样卷积层（转置卷积），长宽double
def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=2, padding=1),
        nn.ReLU(inplace=True)
    )


class PoseExpNet(nn.Module):

    def __init__(self, nb_ref_imgs=2, output_exp=False):
        super(PoseExpNet, self).__init__()
        self.nb_ref_imgs = nb_ref_imgs
        self.output_exp = output_exp

        conv_planes = [16, 32, 64, 128, 256, 256, 256]
        # 第一层卷积层的输入通道数为3*（1+参照图的数量），输出通道数为16
        self.conv1 = conv(3*(1+self.nb_ref_imgs), conv_planes[0], kernel_size=7) # 3*(1+2)->16
        self.conv2 = conv(conv_planes[0], conv_planes[1], kernel_size=5) # 16->32
        self.conv3 = conv(conv_planes[1], conv_planes[2]) # 32->64
        self.conv4 = conv(conv_planes[2], conv_planes[3]) # 64->128
        self.conv5 = conv(conv_planes[3], conv_planes[4]) # 128->256
        self.conv6 = conv(conv_planes[4], conv_planes[5]) # 256->256
        self.conv7 = conv(conv_planes[5], conv_planes[6]) # 256->256

        # 256->6*2 1*1的卷积核，其实也就是全连接层（通道维度的全连接）
        self.pose_pred = nn.Conv2d(conv_planes[6], 6*self.nb_ref_imgs, kernel_size=1, padding=0)

        if self.output_exp:
            # 如果输出可解释性掩模
            # 通过上采样层恢复原来的形状
            upconv_planes = [256, 128, 64, 32, 16]
            self.upconv5 = upconv(conv_planes[4],   upconv_planes[0])
            self.upconv4 = upconv(upconv_planes[0], upconv_planes[1])
            self.upconv3 = upconv(upconv_planes[1], upconv_planes[2])
            self.upconv2 = upconv(upconv_planes[2], upconv_planes[3])
            self.upconv1 = upconv(upconv_planes[3], upconv_planes[4])

            # 跟深度图类似，有四种分辨率的可解释性掩模，padding=k//2，所以形状不变
            self.predict_mask4 = nn.Conv2d(upconv_planes[1], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask3 = nn.Conv2d(upconv_planes[2], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask2 = nn.Conv2d(upconv_planes[3], self.nb_ref_imgs, kernel_size=3, padding=1)
            self.predict_mask1 = nn.Conv2d(upconv_planes[4], self.nb_ref_imgs, kernel_size=3, padding=1)

    def init_weights(self):
        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                xavier_uniform_(m.weight.data) # 采用xavier_uniform_初始化
                if m.bias is not None: # 偏置采用0
                    zeros_(m.bias)

    def forward(self, target_image, ref_imgs):
        assert(len(ref_imgs) == self.nb_ref_imgs)
        input = [target_image]
        input.extend(ref_imgs)
        input = torch.cat(input, 1) # 直接将tgt和两张source沿通道进行拼接
        out_conv1 = self.conv1(input)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)
        out_conv5 = self.conv5(out_conv4)
        out_conv6 = self.conv6(out_conv5)
        out_conv7 = self.conv7(out_conv6)

        pose = self.pose_pred(out_conv7) # 输出pose （1*12*1*4）
        pose = pose.mean(3).mean(2) # 先沿着第三维求平均 (1*12*1*4)->(1*12*1) 再沿着第二维(1*12*1)->(1*12)
        pose = 0.01 * pose.view(pose.size(0), self.nb_ref_imgs, 6) # (1*12)->(1*2*6) 变成两个6维的位姿向量 3维位移加上3维旋转

        if self.output_exp:
            # 可解释性掩模，上采样然后截断，形状保持为第四层卷积层输出大小，就是逐步还原大小的过程，最后变为输入的大小
            out_upconv5 = self.upconv5(out_conv5  )[:, :, 0:out_conv4.size(2), 0:out_conv4.size(3)]
            out_upconv4 = self.upconv4(out_upconv5)[:, :, 0:out_conv3.size(2), 0:out_conv3.size(3)]
            out_upconv3 = self.upconv3(out_upconv4)[:, :, 0:out_conv2.size(2), 0:out_conv2.size(3)]
            out_upconv2 = self.upconv2(out_upconv3)[:, :, 0:out_conv1.size(2), 0:out_conv1.size(3)]
            out_upconv1 = self.upconv1(out_upconv2)[:, :, 0:input.size(2), 0:input.size(3)]

            exp_mask4 = sigmoid(self.predict_mask4(out_upconv4)) # 通过sigmoid函数，输出“概率”
            exp_mask3 = sigmoid(self.predict_mask3(out_upconv3))
            exp_mask2 = sigmoid(self.predict_mask2(out_upconv2))
            exp_mask1 = sigmoid(self.predict_mask1(out_upconv1))
        else:
            exp_mask4 = None
            exp_mask3 = None
            exp_mask2 = None
            exp_mask1 = None

        if self.training:
            return [exp_mask1, exp_mask2, exp_mask3, exp_mask4], pose
        else:
            return exp_mask1, pose
