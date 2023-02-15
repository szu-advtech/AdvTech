import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from CostRegNet import CostReg, Bottleneck


class FeatureNet(nn.Module):    #特征提取网络
    def __init__(self):
        super(FeatureNet, self).__init__()
        self.expanse_ratio = 3
        self.inplanes = 8

        self.conv0 = ConvBnReLU(3, 8, 3, 1, 1)
        self.conv1 = ConvBnReLU(8, 8, 3, 1, 1)

        self.conv2 = ConvBnReLU(8, 16, 5, 2, 2)
        self.conv3 = ConvBnReLU(16, 16, 3, 1, 1)
        self.conv4 = ConvBnReLU(16, 16, 3, 1, 1)

        self.conv5 = ConvBnReLU(16, 32, 5, 2, 2)
        self.conv6 = ConvBnReLU(32, 32, 3, 1, 1)
        self.feature = nn.Conv2d(32, 32, 3, 1, 1)

        '''self.firstconv = nn.Sequential(MobileV2_Residual(3, 8, 1, self.expanse_ratio),
                                       nn.ReLU(inplace=True),
                                       MobileV2_Residual(8, 8, 1, self.expanse_ratio),
                                       nn.ReLU(inplace=True)
                                       )

        self.layer1 = self._make_layer(MobileV1_Residual, 16, 3, 2, 1, 1)
        self.layer2 = self._make_layer(MobileV1_Residual, 32, 3, 2, 1, 1)

    def _make_layer(self, block, planes, blocks, stride, pad, dilation):
        downsample = None

        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = [block(self.inplanes, planes, stride, downsample, pad, dilation)]
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, 1, None, pad, dilation))

        return nn.Sequential(*layers)'''

    def forward(self, x):

        x = self.conv1(self.conv0(x))
        x = self.conv4(self.conv3(self.conv2(x)))
        x = self.feature(self.conv6(self.conv5(x)))

        '''x = self.firstconv(x)
        x = self.layer1(x)
        x = self.layer2(x)'''

        return x


class CostRegNet(nn.Module):    #cost volume正则化网络
    def __init__(self):
        super(CostRegNet, self).__init__()
        self.conv0 = ConvBnReLU3D(32, 8)

        self.conv1 = ConvBnReLU3D(8, 16, stride=2)
        self.conv2 = ConvBnReLU3D(16, 16)

        self.conv3 = ConvBnReLU3D(16, 32, stride=2)
        self.conv4 = ConvBnReLU3D(32, 32)

        self.conv5 = ConvBnReLU3D(32, 64, stride=2)
        self.conv6 = ConvBnReLU3D(64, 64)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)

        '''self.expanse_ratio = 2

        self.conv0 = MobileV2_Residual_3D(32, 8, 1, self.expanse_ratio)

        self.conv1 = MobileV2_Residual_3D(8, 16, 2, self.expanse_ratio)
        self.conv2 = MobileV2_Residual_3D(16, 16, 1, self.expanse_ratio)

        self.conv3 = MobileV2_Residual_3D(16, 32, 2, self.expanse_ratio)
        self.conv4 = MobileV2_Residual_3D(32, 32, 1, self.expanse_ratio)

        self.conv5 = MobileV2_Residual_3D(32, 64, 2, self.expanse_ratio)
        self.conv6 = MobileV2_Residual_3D(64, 64, 1, self.expanse_ratio)

        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(64, 32, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True))

        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(16, 8, 3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True))

        self.prob = nn.Conv3d(8, 1, 3, stride=1, padding=1)'''

    def forward(self, x):

        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv8(x)
        x = conv0 + self.conv9(x)
        x = self.prob(x)

        return x


class RefineNet(nn.Module):    #深度图边缘优化残差网络
    def __init__(self):
        super(RefineNet, self).__init__()
        self.conv1 = ConvBnReLU(4, 32)
        self.conv2 = ConvBnReLU(32, 32)
        self.conv3 = ConvBnReLU(32, 32)
        self.res = ConvBnReLU(32, 1)

        '''self.expanse_ratio = 3
        self.conv1 = nn.Sequential(MobileV2_Residual(4, 32, 1, self.expanse_ratio),
                      nn.ReLU(inplace=True)
                      )
        self.conv2 = nn.Sequential(MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                   nn.ReLU(inplace=True)
                                   )
        self.conv3 = nn.Sequential(MobileV2_Residual(32, 32, 1, self.expanse_ratio),
                                   nn.ReLU(inplace=True)
                                   )
        self.res = nn.Sequential(MobileV2_Residual(32, 1, 1, self.expanse_ratio),
                                   nn.ReLU(inplace=True)
                                   )'''

    def forward(self, img, depth_init):
        concat = F.cat((img, depth_init), dim=1)
        depth_residual = self.res(self.conv3(self.conv2(self.conv1(concat))))
        depth_refined = depth_init + depth_residual
        return depth_refined


class MVSNet(nn.Module):
    def __init__(self, refine=True, cfg=None):
        super(MVSNet, self).__init__()
        self.refine = refine
        self.feature = FeatureNet()
        #self.cost_regularization = CostRegNet()
        self.cost_regularization = CostReg(cfg)

        if self.refine:
            self.refine_network = RefineNet()

    def forward(self, imgs, proj_matrices, depth_values, search=False, **kwargs):
        imgs = torch.unbind(imgs, 1)       #对某一个维度进行长度为1的切片，并将所有切片结果返回。
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"
        img_height, img_width = imgs[0].shape[2], imgs[0].shape[3]
        num_depth = depth_values.shape[1]
        num_views = len(imgs)

        # step 1. feature extraction
        # in: images; out: 32-channel feature maps
        features = [self.feature(img) for img in imgs]
        ref_feature, src_features = features[0], features[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]

        # step 2. differentiable homograph, build cost volume
        ref_volume = ref_feature.unsqueeze(2).repeat(1, 1, num_depth, 1, 1)
        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2
        del ref_volume
        for src_fea, src_proj in zip(src_features, src_projs):
            # warpped features
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_values)
            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)  # the memory of warped_volume has been modified
            del warped_volume
        # aggregate multiple feature volumes by variance
        volume_variance = volume_sq_sum.div_(num_views).sub_(volume_sum.div_(num_views).pow_(2))

        # step 3. cost volume regularization
        cost_reg = self.cost_regularization(volume_variance, search)
        # cost_reg = F.upsample(cost_reg, [num_depth * 4, img_height, img_width], mode='trilinear')
        cost_reg = cost_reg.squeeze(1)
        prob_volume = F.softmax(cost_reg, dim=1)
        depth = depth_regression(prob_volume, depth_values=depth_values)

        with torch.no_grad():
            # photometric confidence
            prob_volume_sum4 = 4 * F.avg_pool3d(F.pad(prob_volume.unsqueeze(1), pad=(0, 0, 0, 0, 1, 2)), (4, 1, 1), stride=1, padding=0).squeeze(1)
            depth_index = depth_regression(prob_volume, depth_values=torch.arange(num_depth, device=prob_volume.device, dtype=torch.float)).long()
            photometric_confidence = torch.gather(prob_volume_sum4, 1, depth_index.unsqueeze(1)).squeeze(1)

        # step 4. depth map refinement
        if not self.refine:
            return {"depth": depth, "photometric_confidence": photometric_confidence}
        else:
            refined_depth = self.refine_network(torch.cat((imgs[0], depth), 1))
            return {"depth": depth, "refined_depth": refined_depth, "photometric_confidence": photometric_confidence}


def mvsnet_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)