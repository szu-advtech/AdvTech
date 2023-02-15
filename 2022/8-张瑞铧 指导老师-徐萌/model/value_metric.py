import math
from skimage.metrics import structural_similarity as ssim
import torch


# MAE平均绝对误差
def cloud_mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_pred - y_true))


# MSE均方误差
def cloud_mean_squared_error(y_true, y_pred):
    return torch.mean(torch.square(y_pred - y_true))


# RMSE均方根差
def cloud_root_mean_squared_error(y_true, y_pred):
    return torch.sqrt(torch.mean(torch.square(y_pred - y_true)))


# SAM 光谱角
def get_sam(y_true, y_predict):
    mat = torch.multiply(y_true, y_predict)
    mat = torch.sum(mat, dim=1)
    mat = torch.div(mat, torch.sqrt(torch.sum(torch.multiply(y_true, y_true), dim=1)))
    mat = torch.div(mat, torch.sqrt(torch.sum(torch.multiply(y_predict, y_predict), dim=1)))
    mat = torch.acos(torch.clip(mat, -1, 1))

    return mat


# 计算全图的SAM
def cloud_mean_sam(y_true, y_predict):
    mat = get_sam(y_true, y_predict)

    return torch.mean(mat)


# PSNR 峰值信噪比
def cloud_psnr(y_true, y_predict):
    y_true *= 2000
    y_predict *= 2000
    rmse = torch.sqrt(torch.mean(torch.square(y_predict[:, 0:13, :, :] - y_true[:, 0:13, :, :])))

    return 20.0 * (torch.log(10000.0 / rmse) / math.log(10.0))


# 计算SSIM
def get_ssim(y_true, y_predict):
    return ssim((torch.squeeze(y_true)).cpu().numpy(), (torch.squeeze(y_predict)).cpu().numpy(), channel_axis=True)


# CARL损失函数
def carl_error(y_true, y_predict, cloud_shadow_mask, Lambda=1.0):
    clearmask = torch.ones_like(cloud_shadow_mask) - cloud_shadow_mask
    predicted = y_predict
    input_cloudy = y_predict
    target = y_true

    cscmae = torch.mean(clearmask * torch.abs(predicted - input_cloudy) + cloud_shadow_mask * torch.abs(
        predicted - target)) + Lambda * torch.mean(torch.abs(predicted - target))

    return cscmae







def cloud_mean_absolute_error_clear(y_true, y_pred, x_input):
    """Computes the SAM over the clear image parts."""
    clearmask = torch.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred
    input_cloudy = x_input[:, 2:, :, :]

    if torch.sum(clearmask) == 0:
        return 0.0

    clti = clearmask * torch.abs(predicted - input_cloudy)
    clti = torch.sum(clti) / (torch.sum(clearmask) * 13)

    return clti


def cloud_mean_absolute_error_covered(y_true, y_pred):
    """Computes the SAM over the covered image parts."""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    predicted = y_pred
    target = y_true

    if torch.sum(cloud_cloudshadow_mask) == 0:
        return 0.0

    ccmaec = cloud_cloudshadow_mask * torch.abs(predicted - target)
    ccmaec = torch.sum(ccmaec) / (torch.sum(cloud_cloudshadow_mask) * 13)

    return ccmaec


def cloud_mean_sam_covered(y_true, y_pred):
    """Computes the SAM over the covered image parts."""
    cloud_cloudshadow_mask = y_true[:, -1:, :, :]
    target = y_true[:, 0:13, :, :]
    predicted = y_pred[:, 0:13, :, :]

    if torch.sum(cloud_cloudshadow_mask) == 0:
        return 0.0

    sam = get_sam(target, predicted)
    sam = torch.unsqueeze(sam, dim=1)
    sam = torch.sum(cloud_cloudshadow_mask * sam) / torch.sum(cloud_cloudshadow_mask)

    return sam


def cloud_mean_sam_clear(y_true, y_pred):
    """Computes the SAM over the clear image parts."""
    clearmask = torch.ones_like(y_true[:, -1:, :, :]) - y_true[:, -1:, :, :]
    predicted = y_pred[:, 0:13, :, :]
    input_cloudy = y_pred[:, -14:-1, :, :]

    if torch.sum(clearmask) == 0:
        return 0.0

    sam = get_sam(input_cloudy, predicted)
    sam = torch.unsqueeze(sam, dim=1)
    sam = torch.sum(clearmask * sam) / torch.sum(clearmask)

    return sam