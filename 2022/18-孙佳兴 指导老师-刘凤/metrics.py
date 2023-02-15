# -*- coding: utf-8 -*-
"""
License: GNU-3.0
Code Reference:https://github.com/wasaCheney/IQA_pansharpening_python
"""
import math

import numpy as np


def SAM(HS_fusion, gtHS):
    """
    :param HS_fusion: 预测数据
    :param gtHS: 真实数据
    :return: 所有像点的SAM均值
    """
    assert gtHS.ndim == 3 and gtHS.shape == HS_fusion.shape

    dot_sum = np.sum(gtHS * HS_fusion, axis=2)
    norm_true = np.linalg.norm(gtHS, axis=2)
    norm_pred = np.linalg.norm(HS_fusion, axis=2)

    res = np.arccos(dot_sum / norm_pred / norm_true)
    is_nan = np.nonzero(np.isnan(res))
    for (x, y) in zip(is_nan[0], is_nan[1]):
        res[x, y] = 0
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            res[i, j] = math.degrees(res[i, j])

    sam = np.mean(res)
    return sam


def rmse(HS_fusion, gtHS):
    """
    :param HS_fusion:
    :param gtHS:
    :return:
    """
    if len(HS_fusion.shape) == 3:
        channels = HS_fusion.shape[2]
    else:
        channels = 1
        HS_fusion = np.reshape(HS_fusion, (HS_fusion.shape[0], HS_fusion.shape[1], 1))
        gtHS = np.reshape(gtHS, (gtHS.shape[0], gtHS.shape[1], 1))
    HS_fusion = HS_fusion.astype(np.float32)
    gtHS = gtHS.astype(np.float32)

    def single_rmse(img1, img2):
        diff = img1 - img2
        mse = np.mean(np.square(diff))
        return np.sqrt(mse)

    rmse_sum = 0
    for band in range(channels):
        fake_band_img = HS_fusion[:, :, band]
        real_band_img = gtHS[:, :, band]
        rmse_sum += single_rmse(fake_band_img, real_band_img)

    rmse = round(rmse_sum, 2)

    return rmse


def ERGAS(HS_fusion, gtHS, LRHS):
    """

    :param HS_fusion: 预测数据
    :param gtHS: 真实数据
    :param LRHS: 原始数据
    :return:
    """
    ratio = gtHS.shape[0] / LRHS.shape[0]  # 低分辨率影像空间分辨率和高分辨率影像空间分辨率的比值
    # 此处也可通过列数计算，此处只是完全按照定义来看

    channels = HS_fusion.shape[2]  # 通道数

    inner_sum = 0
    for channel in range(channels):
        band_img1 = HS_fusion[:, :, channel]
        band_img2 = gtHS[:, :, channel]
        band_img3 = LRHS[:, :, channel]

        rmse_value = rmse(band_img1, band_img2)
        m = np.mean(band_img3)
        inner_sum += np.power((rmse_value / m), 2)
    mean_sum = inner_sum / channels
    ergas = 100 * np.sqrt(mean_sum) / ratio

    return ergas


def scc(img1, img2):
    """SCC for 2D (H, W)or 3D (H, W, C) image; uint or float[0, 1]"""
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    img1_ = img1.astype(np.float64)
    img2_ = img2.astype(np.float64)
    if img1_.ndim == 2:
        return np.corrcoef(img1_.reshape(1, -1), img2_.rehshape(1, -1))[0, 1]
    elif img1_.ndim == 3:
        # print(img1_[..., i].reshape[1, -1].shape)
        # test = np.corrcoef(img1_[..., i].reshape[1, -1], img2_[..., i].rehshape(1, -1))
        # print(type(test))
        ccs = [np.corrcoef(img1_[..., i].reshape(1, -1), img2_[..., i].reshape(1, -1))[0, 1]
               for i in range(img1_.shape[2])]
        return np.mean(ccs)
    else:
        raise ValueError('Wrong input image dimensions.')


def ref_evaluate(pred, gt, lrhs):
    # reference metrics
    c_scc = scc(pred, gt)
    c_sam = SAM(pred, gt)
    c_rmse = rmse(pred, gt)
    c_ergas = ERGAS(pred, gt, lrhs)

    return [c_scc, c_sam, c_rmse, c_ergas]
