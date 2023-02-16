from collections import OrderedDict
import torch
import torch.nn as nn
import random
import h5py
import numpy as np
from torchvision import models, transforms
from torch.nn import init
from torch.nn import functional as F
from scipy.signal import convolve2d
import cv2

# class VGG19(nn.Module):
#     def __init__(self):
#         super(VGG19, self).__init__()
#         '''
#          use vgg19 conv1_2, conv2_2, conv3_3 feature, before relu layer
#         '''
#         self.feature_list = [2, 7, 14]
#         vgg19 = torchvision.models.vgg19(pretrained=True)
#
#         self.model = torch.nn.Sequential(*list(vgg19.features.children())[:self.feature_list[-1]+1])
#
#     def forward(self, x):
#         x = (x-0.5)/0.5
#         features = []
#         for i, layer in enumerate(list(self.model)):
#             x = layer(x)
#             if i in self.feature_list:
#                 features.append(x)
#         return features
#
# def VGGPerceptualLoss(fakeIm, realIm, vggnet):
#     weights = [1, 0.2, 0.04]
#     features_fake = vggnet(fakeIm)
#     features_real = vggnet(realIm)
#     features_real_no_grad = [f_real.detach() for f_real in features_real]
#     mse_loss = nn.MSELoss(reduction='elementwise_mean')
#
#     loss = 0
#     for i in range(len(features_real)):
#         loss_i = mse_loss(features_fake[i], features_real_no_grad[i])
#         loss = loss + loss_i * weights[i]
#
#     return loss

def split_img(img_path, img_hw, overlap, is_test=False, is_rgb = False, img_name = None):
    """
        先把img由mat转成np_array，然后再进行操作，保存成.npy格式文件方便之后的操作
        i代表纵向偏移（行）
        j代表横向偏移（列）
    """
    if is_rgb:
        img = cv2.imread(img_path)
        img_shape = img.shape

    else:
        mat = h5py.File(img_path)
        img = np.transpose(np.array(mat[img_name]))
        img_shape = img.shape

    if not is_test:
        list_train_data = []
        i = 0
        while i < img_shape[0]:  # 行
            if (i + img_hw) < img_shape[0]:
                j = 0
                while j < img_shape[1]:  # 列
                    if (j + img_hw) < img_shape[1]:
                        temp_img = img[i:i + img_hw, j:j + img_hw, :]
                        list_train_data.append(temp_img)
                        j += overlap
                    elif (j + img_hw) > img_shape[1]:
                        list_train_data.append(img[i:i + img_hw, img_shape[1] - img_hw:img_shape[1], :])
                        j += overlap
                i += overlap

            elif (i + img_hw) > img_shape[0]:
                j = 0
                while j < img_shape[1]:
                    if (j + img_hw) < img_shape[1]:
                        list_train_data.append(img[img_shape[0] - img_hw:img_shape[0], j:j + img_hw, :])
                        j += overlap
                    elif (j + img_hw) > img_shape[1]:
                        list_train_data.append(
                            img[img_shape[0] - img_hw:img_shape[0], img_shape[1] - img_hw:img_shape[1], :])
                        j += overlap
                i += overlap

    else:
        int_x = random.randint(img_hw, img_shape[0] - img_hw)
        int_y = random.randint(img_hw, img_shape[1] - img_hw)
        test_img = img[int_x: int_x + img_hw, int_y: int_y + img_hw, :]
        i = 0
        j = 0
        list_train_data = []
        while i < img_shape[0]:
            # i在测试图像上边
            if (i + img_hw) <= int_x:
                while j < img_shape[1]:
                    if j + img_hw < img_shape[1]:
                        list_train_data.append(img[i: i + img_hw, j: j + img_hw, :])
                        j += overlap
                    else:
                        list_train_data.append(img[i: i + img_hw, img_shape[1] - img_hw: img_shape[1], :])
                        j += overlap
                        break
                j = 0
                i += overlap

            # i在测试图像下面
            elif i >= (int_x + img_hw) and (i + img_hw) < img_shape[0]:
                while j < img_shape[1]:
                    if j + img_hw < img_shape[1]:
                        list_train_data.append(img[i: i + img_hw, j: j + img_hw, :])
                        j += overlap
                    else:
                        list_train_data.append(img[i: i + img_hw, img_shape[1] - img_hw: img_shape[1], :])
                        j += overlap
                        break
                j = 0
                i += overlap

            # i在测试图像中间
            elif int_x <= i <= int_x + img_hw:
                if (i + img_hw) < img_shape[0]:  # 没超过边界
                    while j < img_shape[1]:
                        if (j + img_hw) <= int_y:
                            list_train_data.append(img[i: i + img_hw, j: j + img_hw, :])
                            j += overlap

                        elif int_y < (j + img_hw) <= (int_y + img_hw):
                            list_train_data.append(img[i: i + img_hw, int_y - img_hw: int_y, :])
                            j = int_y + img_hw

                        elif j > int_y and (j + img_hw) < img_shape[1]:
                            list_train_data.append(img[i: i + img_hw, j: j + img_hw, :])
                            j += overlap

                        elif (j + img_hw) > img_shape[1]:
                            list_train_data.append(img[i: i + img_hw, img_shape[1] - img_hw: img_shape[1], :])
                            break
                    j = 0
                    i += overlap

                else:  # 超过边界
                    while j < img_shape[1]:
                        if (j + img_hw) <= int_y:
                            list_train_data.append(img[img_shape[0] - img_hw: img_shape[0], j: j + img_hw, :])
                            j += overlap

                        elif int_y < (j + img_hw) <= (int_y + img_hw):
                            list_train_data.append(img[img_shape[0] - img_hw: img_shape[0], int_y - img_hw: int_y, :])
                            j = int_y + img_hw

                        elif j > int_y and (j + img_hw) < img_shape[1]:
                            list_train_data.append(img[img_shape[0] - img_hw: img_shape[0], j: j + img_hw, :])
                            j += overlap

                        elif (j + img_hw) > img_shape[1]:
                            list_train_data.append(
                                img[img_shape[0] - img_hw: img_shape[0], img_shape[1] - img_hw: img_shape[1], :])
                            break
                    j = 0
                    i += overlap


            # i在测试图像上中
            elif int_x < (i + img_hw) <= (int_x + img_hw):
                while j < img_shape[1]:
                    if (j + img_hw) <= int_y:
                        list_train_data.append(img[i: i + img_hw, j: j + img_hw, :])
                        j += overlap

                    elif int_y < (j + img_hw) <= (int_y + img_hw):
                        list_train_data.append(img[int_x - img_hw: int_x, int_y - img_hw: int_y, :])
                        j = int_y + img_hw

                    elif j > int_y and (j + img_hw) < img_shape[1]:
                        list_train_data.append(img[i: i + img_hw, j: j + img_hw, :])
                        j += overlap

                    elif (j + img_hw) > img_shape[1]:
                        list_train_data.append(img[i: i + img_hw, img_shape[1] - img_hw: img_shape[1], :])
                        break
                j = 0
                i += overlap

            # i + Offset 超出边界
            elif (i + img_hw) >= img_shape[0]:
                while j < img_shape[1]:
                    if j + img_hw < img_shape[1]:
                        list_train_data.append(img[img_shape[0] - img_hw: img_shape[0], j: j + img_hw, :])
                        j += overlap
                    else:
                        list_train_data.append(
                            img[img_shape[0] - img_hw: img_shape[0], img_shape[1] - img_hw: img_shape[1], :])
                        j += overlap
                        break
                break


    if is_test:
        return list_train_data, test_img
    else:
        return list_train_data


class TransformLR:
    def __init__(self, optimizer, epoch=0, initial_lr=0.0001):
        self.optimizer = optimizer
        self.epoch = epoch
        self.num_step = 0
        self.step()
        self.initial_lr = initial_lr

    def step(self):
        if self.epoch // 2000 == 0:
            new_lr = self.initial_lr * 0.1
            for group in self.optimizer.param_groups:
                group['lr'] = new_lr
            self.num_step += 1

        else:
            pass


def loss_function(predict, ground_truth, mode):
    if mode == 'l1_loss':
        return F.l1_loss(predict, ground_truth)

    elif mode == 'l2_loss':
        return F.mse_loss(predict, ground_truth)

    elif mode == 'sam_loss':
        return sam(predict, ground_truth)


# class L1_loss():
#     def __init__(self):
#         super(L1_loss, self).__init__()
#         pass
#
#     def forward(self, predict, ground_truth):
#         # tensor_one = torch.ones_like(predict)
#         # loss = torch.tensor(abs(predict - ground_truth).sum() / tensor_one.sum(),requires_grad = True)
#         loss = F.l1_loss(predict, ground_truth)
#         return loss
#
#
# class L2_loss(nn.Module):
#     def __init__(self):
#         super(L2_loss, self).__init__()
#         pass
#
#     def forward(self, predict, ground_truth):
#         """
#
#         Returns
#         -------
#         tensor
#         """
#         # tensor_one = torch.ones_like(predict)
#         # loss = torch.pow(predict - ground_truth,2).sum() / tensor_one.sum()
#         loss = F.mse_loss(predict, ground_truth)
#         return loss
#
#
# class SAM_loss(nn.Module):
#     def __init__(self):
#         super(SAM_loss, self).__init__()
#         pass
#
#     def forward(self, predict, ground_truth):
#         return sam(predict, ground_truth)


# Deep_Learning on IQA(Image Quality Assessment)
class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2) // 2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:, None] * a[None, :])
        g = g / torch.sum(g)
        self.register_buffer('filter', g[None, None, :, :].repeat((self.channels, 1, 1, 1)))

    def forward(self, x):
        x = x ** 2
        out = F.conv2d(x, self.filter, stride=self.stride, padding=self.padding, groups=x.shape[1])
        return (out + 1e-12).sqrt()


# ---------------------------------------IQA by deep_learning------------------------------------------
class DISTS(torch.nn.Module):
    def __init__(self, spectrum_band):
        super(DISTS, self).__init__()
        self.spectrum_band = spectrum_band
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        self.stage1.add_module(str(0), nn.Conv2d(self.spectrum_band, 64, 3, 1))
        for x in range(1, 4):  # 第一个卷积核用随机初始化
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False
        # self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        # self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [self.spectrum_band, 64, 128, 256, 512, 512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns), 1, 1)))
        self.alpha.data.normal_(0.1, 0.01)
        self.beta.data.normal_(0.1, 0.01)
        # if load_weights:  # weight文件要改变，把前面3个数的值改成128个数
        #     weights = torch.load(os.path.join(sys.prefix,'weights.pt'))
        #     self.alpha.data = weights['alpha']
        #     self.beta.data = weights['beta']

    def forward_once(self, x):
        h = x.float()
        # h = (h-h.mean)/h.std  # 这个地方你看要不要

        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x, h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, require_grad=False, batch_average=False):  # x，y输入的形式是（batch，c，h，w）
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        dist1 = 0
        dist2 = 0
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha / w_sum, self.chns, dim=1)
        beta = torch.split(self.beta / w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2, 3], keepdim=True)
            y_mean = feats1[k].mean([2, 3], keepdim=True)
            S1 = (2 * x_mean * y_mean + c1) / (x_mean ** 2 + y_mean ** 2 + c1)
            dist1 = dist1 + (alpha[k] * S1).sum(1, keepdim=True)

            x_var = ((feats0[k] - x_mean) ** 2).mean([2, 3], keepdim=True)
            y_var = ((feats1[k] - y_mean) ** 2).mean([2, 3], keepdim=True)
            xy_cov = (feats0[k] * feats1[k]).mean([2, 3], keepdim=True) - x_mean * y_mean
            S2 = (2 * xy_cov + c2) / (x_var + y_var + c2)
            dist2 = dist2 + (beta[k] * S2).sum(1, keepdim=True)

        score = 1 - (dist1 + dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score


def prepare_image(image, resize=False):
    if resize and min(image.size) > 256:
        image = transforms.functional.resize(image, 256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)


def weight_init(m):
    global TAG
    print(m)
    if isinstance(m, nn.Conv2d) and TAG == 0:
        init.xavier_uniform(m.weight)
        init.constant(m.bias, 0)
        TAG = 1
    elif isinstance(m, nn.BatchNorm2d):
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)
    elif isinstance(m, nn.Linear):
        init.normal(m.weight, std=1e-3)
        if m.bias:
            init.constant(m.bias, 0)


# -----------------------------------------------------------------------------------------------------
# metric
# Spectral-Angle-Mapper (SAM)
def sam(H_fuse, H_ref, epsilon=1e-6):
    # Compute number of spectral bands
    H_ref, H_fuse = H_ref.squeeze(), H_fuse.squeeze()

    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse = H_fuse.reshape(N_spectral, -1)
    H_ref = H_ref.reshape(N_spectral, -1)
    N_pixels = H_fuse.shape[1]

    # Calculating inner product
    inner_prod = torch.nansum(H_fuse * H_ref, 0)
    fuse_norm = (torch.nansum(H_fuse ** 2, dim=0) + epsilon).sqrt()
    ref_norm = (torch.nansum(H_ref ** 2, dim=0) + epsilon).sqrt()

    # Calculating SAM
    SAM = torch.rad2deg(torch.nansum(torch.acos(inner_prod / (fuse_norm * ref_norm)) + epsilon) / N_pixels)
    return SAM


# Root-Mean-Squared Error (RMSE)
def rmse(H_fuse, H_ref):
    # Rehsaping fused and reference data
    H_ref, H_fuse = H_ref.squeeze(), H_fuse.squeeze()
    H_fuse_reshaped = H_fuse.reshape(-1).to('cpu')
    H_ref_reshaped = H_ref.reshape(-1).to('cpu')

    # Calculating RMSE
    RMSE = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped) ** 2) / H_fuse_reshaped.shape[0])
    return RMSE


# Error Relative Global A dimensional Syntheses (ERGAS)
def egras(H_fuse, H_ref, scale_factor):
    # Compute number of spectral bands
    H_ref, H_fuse = H_ref.squeeze(), H_fuse.squeeze()

    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse = H_fuse.reshape(N_spectral, -1).to('cpu')
    H_ref = H_ref.reshape(N_spectral, -1).to('cpu')
    N_pixels = H_fuse.shape[1]

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.nansum((H_ref - H_fuse) ** 2, dim=1) / N_pixels)
    mu_ref = torch.mean(H_ref, dim=1)

    # Calculating Error Relative Global A dimensional Syntheses (ERGAS)
    ergas = 100 * (1 / scale_factor ** 2) * torch.sqrt(torch.nansum(torch.div(rmse, mu_ref) ** 2) / N_spectral)
    return ergas


# Peak SNR (PSNR)
def pnsr(H_fuse, H_ref):
    # Compute number of spectral bands
    if H_ref.dim() == 4 and H_fuse.dim() == 4:
        pass
    else:
        H_ref, H_fuse = H_ref.squeeze(), H_fuse.squeeze()

    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse = H_fuse.reshape(N_spectral, -1).to('cpu')
    H_ref = H_ref.reshape(N_spectral, -1).to('cpu')

    # Calculating RMSE of each band
    rmse = torch.sum((H_ref - H_fuse) ** 2, dim=1) / H_fuse.shape[1]

    # Calculating max of H_ref for each band
    max_H_ref, _ = torch.max(H_ref, dim=1)

    # Calculating PSNR
    pnsr = torch.nansum(10 * torch.log10(torch.div(max_H_ref ** 2, rmse))) / N_spectral

    return pnsr


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=1):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=np.array([win_size, win_size]), sigma=1.5)
    window = window.astype(np.float32) / np.sum(np.sum(window))

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)).astype(np.float32) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


def SSIM(pred, gt):
    ssim = 0
    for i in range(gt.shape[0]):
        ssim = ssim + compute_ssim(pred[i, :, :], gt[i, :, :])
    return ssim / gt.shape[0]


# 截取模型中的某些层
class Intermediate_layer_getter(nn.ModuleDict):
    """ get the output of certain layers """

    def __init__(self, model, return_layers):
        # 判断传入的return_layers是否存在于model中
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")

        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}  # 构造dict
        layers = OrderedDict()
        # 将要从model中获取信息的最后一层之前的模块全部复制下来
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super(Intermediate_layer_getter, self).__init__(layers)  # 将所需的网络层通过继承的方式保存下来
        self.return_layers = orig_return_layers

    def forward(self, x):
        out = OrderedDict()
        # 将所需的值以k,v的形式保存到out中
        for name, module in self.named_children():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out
