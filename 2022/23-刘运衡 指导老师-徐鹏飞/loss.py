from cv2 import log
import torch
import torch.nn as nn
import torch.nn.functional as func
from torchvision.models import vgg16, VGG16_Weights
from modules import Render
import utils

# 自定义advModel
class advModel(nn.Module):
    def __init__(self):
        super(advModel, self).__init__()
        # 定义全局判别器，输出为两个分类结果
        self.classifierxDG = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1296, 1),
            nn.Sigmoid()
        )
        # 定义局部判别器，输出为两个分类结果
        self.classifierxDL = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1296, 1),
            nn.Sigmoid()
        )
        ################################################################
        # 定义VGG16的卷积层
        ################################################################
        # 加载VGG16模型
        # self.vgg = vgg16(pretrained=False).eval().cuda()
        self.vgg = vgg16(weights=VGG16_Weights.DEFAULT).eval().cuda()
        # 加载模型参数
        # self.vgg.load_state_dict(torch.load('./vgg16-397923af.pth'))
        # 注册hook函数，获取模型的“conv3_pool”层
        features = list(self.vgg.children())[0]
        hook_layer = features[16]
        hook_layer.register_forward_hook(self.hookFeature)
        # 卷积块，激活函数为ReLU，卷积核大小为3*3，输出通道数为64
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=1,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, y):
        # 全局判别器
        xDG = 0
        # # VGG16的卷积层
        self.vgg(x)
        # # 卷积层的输出
        self.vgg16_feature = self.conv(self.vgg16_feature)
        # # 全局判别器
        xDG = self.classifierxDG(self.vgg16_feature)
        # 局部判别器
        # 获取遮罩后的图像
        x = self.mask(x) * x
        # 局部判别器
        xDL = self.classifierxDL(self.vgg16_feature)
        # 返回结果
        return [xDG, xDL]

    # hook函数，获取中间层的特征图输出
    def hookFeature(self, module, inp, outp):
        # print('hook called')
        self.vgg16_feature = outp

    # 用来对图像产生遮罩
    def mask(self, x):
        # 计算像素点的平均值
        mean = torch.mean(x)
        # 将0.92*mean的值作为阈值过滤
        mask = torch.where(mean > 0.92 * mean,
                           torch.ones_like(mean), torch.zeros_like(mean))
        # 返回遮罩
        return mask


class SVBRDFL1Loss(nn.Module):
    def __init__(self, less_channel=False, use_log=True):
        super(SVBRDFL1Loss, self).__init__()
        # 计算loss是否使用对数变换
        self.use_log = use_log
        self.less_channel = less_channel

    def forward(self, input, target):

        input_normals,  input_diffuse,  input_roughness,  input_specular = utils.unpack_svbrdf(
            input, less_channel=self.less_channel)
        target_normals, target_diffuse, target_roughness, target_specular = utils.unpack_svbrdf(
            target, less_channel=self.less_channel)

        # 当使用角度表示时范围为[0,1]
        # input_normals = torch.clamp(input_normals, 0 if self.less_channel else -1, 1)
        input_normals = torch.clamp(input_normals, 0, 1)
        input_diffuse = torch.clamp(input_diffuse, 0, 1)
        input_roughness = torch.clamp(input_roughness, 0, 1)
        input_specular = torch.clamp(input_specular, 0, 1)

        if self.use_log:
            epsilon_l1 = 1e-5  # 0.01
            input_diffuse = torch.log(input_diffuse + epsilon_l1)
            input_specular = torch.log(input_specular + epsilon_l1)
            target_diffuse = torch.log(target_diffuse + epsilon_l1)
            target_specular = torch.log(target_specular + epsilon_l1)

        return nn.functional.l1_loss(input_normals, target_normals) + nn.functional.l1_loss(input_diffuse, target_diffuse) + nn.functional.l1_loss(input_roughness, target_roughness) + nn.functional.l1_loss(input_specular, target_specular)


class RenderingLoss(nn.Module):
    def __init__(self, less_channel=False, use_log=True):
        super(RenderingLoss, self).__init__()
        # 计算loss是否使用对数变换
        self.use_log = use_log
        self.renderer = Render(less_channel=less_channel)
        self.random_configuration_count = 3
        self.specular_configuration_count = 6

    def forward(self, input, target):
        # B C H W
        batch_size = input.shape[0]

        # TODO 实现和原论文略有区别
        batch_input_renderings = []
        batch_target_renderings = []
        scenes = utils.generate_diffuse_scenes(self.random_configuration_count) + utils.generate_specular_scenes(
            self.specular_configuration_count, size=input.shape[-2:])
        for i in range(len(scenes)):
            batch_input_renderings.append(self.renderer(input, **scenes[i])[0])
            batch_target_renderings.append(
                self.renderer(target, **scenes[i])[0])
        batch_input_renderings = torch.stack(batch_input_renderings, dim=0)
        batch_target_renderings = torch.stack(batch_target_renderings, dim=0)
        if self.use_log:
            epsilon_render = 1e-5  # 0.1
            batch_input_renderings = torch.log(
                batch_input_renderings + epsilon_render)
            batch_target_renderings = torch.log(
                batch_target_renderings + epsilon_render)

        loss = nn.functional.l1_loss(
            batch_input_renderings, batch_target_renderings)

        return loss


class HALoss(nn.Module):
    def __init__(self, renderer=None, less_channel=False, use_log=True):
        super(HALoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.rendering_loss = RenderingLoss(
            less_channel=less_channel, use_log=use_log)
        self.use_log = use_log

    def forward(self, x, y, dis=None):
        # 判断是否有判别器损失
        if dis is not None:
            # 计算目标图像均值
            mean = torch.mean(y)
            dis_loss = 0
            for d in dis:
                # normal & diffuse
                dis_loss += (mean-torch.log(1-d[0])) + \
                    (mean-torch.log(1-d[1])).mean()
        else:
            dis_loss = torch.zeros(1, device=x.device)
        l1 = 0
        if self.use_log:
            l1 = self.l1_loss(torch.log(x+1e-5), torch.log(y+1e-5))
        else:
            l1 = self.l1_loss(x, y)
        return l1, self.rendering_loss(x, y), dis_loss.mean()
