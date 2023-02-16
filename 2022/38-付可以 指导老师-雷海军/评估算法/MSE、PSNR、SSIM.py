# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage import metrics
from scipy import signal


def compute_mse(X, Y):
    """
    compute mean square error of two images

    Parameters:
    -----------
    X, Y: numpy array
        two images data

    Returns:
    --------
    mse: float
        mean square error
    """
    X = np.float32(X)
    Y = np.float32(Y)
    mse = np.mean((X - Y) ** 2, dtype=np.float64)
    return mse


def compute_psnr(X, Y, data_range):
    """
    compute peak signal to noise ratio of two images

    Parameters:
    -----------
    X, Y: numpy array
        two images data

    Returns:
    --------
    psnr: float
        peak signal to noise ratio
    """
    mse = compute_mse(X, Y)
    print("==============mse===============")
    print(mse)
    psnr = 10 * np.log10((data_range ** 2) / mse)
    return psnr


def compute_ssim(X, Y, win_size=7, data_range=None):
    """
    compute structural similarity of two images

    Parameters:
    -----------
    X, Y: numpy array
        two images data
    win_size: int
        window size of image patch for computing ssim of one single position
    data_range: int or float
        maximum dynamic range of image data type

    Returns:
    --------
    mssim: float
        mean structural similarity
    ssim_map: numpy array (float)
        structural similarity map, same shape as input images
    """
    assert X.shape == Y.shape, "X, Y must have same shape"
    assert X.dtype == Y.dtype, "X, Y must have same dtype"
    assert win_size <= np.min(X.shape[0:2]), \
        "win_size should be <= shorter edge of image"
    assert win_size % 2 == 1, "win_size must be odd"
    if data_range is None:
        if 'float' in str(X.dtype):
            data_range = 1
        elif 'uint8' in str(X.dtype):
            data_range = 255
        else:
            raise ValueError(
                'image dtype must be uint8 or float when data_range is None')

    X = np.squeeze(X)
    Y = np.squeeze(Y)
    if X.ndim == 2:
        mssim, ssim_map = _ssim_one_channel(X, Y, win_size, data_range)
    elif X.ndim == 3:
        ssim_map = np.zeros(X.shape)
        for i in range(X.shape[2]):
            _, ssim_map[:, :, i] = _ssim_one_channel(
                X[:, :, i], Y[:, :, i], win_size, data_range)
        mssim = np.mean(ssim_map)
    else:
        raise ValueError("image dimension must be 2 or 3")
    return mssim, ssim_map


def _ssim_one_channel(X, Y, win_size, data_range):
    """
    compute structural similarity of two single channel images

    Parameters:
    -----------
    X, Y: numpy array
        two images data
    win_size: int
        window size of image patch for computing ssim of one single position
    data_range: int or float
        maximum dynamic range of image data type

    Returns:
    --------
    mssim: float
        mean structural similarity
    ssim_map: numpy array (float)
        structural similarity map, same shape as input images
    """
    X, Y = normalize(X, Y, data_range)
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    num = win_size ** 2
    kernel = np.ones([win_size, win_size]) / num
    mean_map_x = convolve2d(X, kernel)
    mean_map_y = convolve2d(Y, kernel)

    mean_map_xx = convolve2d(X * X, kernel)
    mean_map_yy = convolve2d(Y * Y, kernel)
    mean_map_xy = convolve2d(X * Y, kernel)

    cov_norm = num / (num - 1)
    var_x = cov_norm * (mean_map_xx - mean_map_x ** 2)
    var_y = cov_norm * (mean_map_yy - mean_map_y ** 2)
    covar_xy = cov_norm * (mean_map_xy - mean_map_x * mean_map_y)

    A1 = 2 * mean_map_x * mean_map_y + C1
    A2 = 2 * covar_xy + C2
    B1 = mean_map_x ** 2 + mean_map_y ** 2 + C1
    B2 = var_x + var_y + C2

    ssim_map = (A1 * A2) / (B1 * B2)
    mssim = np.mean(ssim_map)
    return mssim, ssim_map


def normalize(X, Y, data_range):
    """
    convert dtype of two images to float64, and then normalize them by
    data_range

    Paramters:
    ----------
    X, Y: numpy array
        two images data
    data_range: int or float
        maximum dynamic range of image data type

    Returns:
    --------
    X, Y: numpy array
        two images
    """
    X = X.astype(np.float64) / data_range
    Y = Y.astype(np.float64) / data_range
    return X, Y


def convolve2d(image, kernel):
    """
    convolve single channel image and kernel

    Parameters:
    -----------
    image: numpy array
        single channel image data
    kernel: numpy array
        kernel data

    Returns:
    --------
    result: numpy array
        image data, same shape as input image
    """
    result = signal.convolve2d(image, kernel, mode='same', boundary='fill')
    return result


def color_to_gray(image, normalization=False):
    """
    convert color image to gray image

    Parameters:
    -----------
    image: numpy array
        color image data
    normalization: bool
        whether to do nomalization

    Returns:
    --------
    image: numpy array
        gray image data
    """
    image = cv2.cvtColor(image, code=cv2.COLOR_BGR2GRAY)
    if normalization and 'uint8' in str(image.dtype):
        image = image.astype(np.float64) / 255
    return image


def resample(image, factor, interpolation):
    """
    down sample and then upsample image by factor using a certain interpolation
    method

    Parameters:
    -----------
    image: numpy array
        image data
    factor:
        sampling factor
    interpolation: enum
        interpolation type

    Returns:
    --------
    image_resample: numpy array
        resampled image data
    """
    height, width = image.shape[0:2]
    downsample_size = (int(width / factor), int(height / factor))
    image_resample = cv2.resize(image,
                                dsize=downsample_size,
                                interpolation=interpolation)
    image_resample = cv2.resize(image_resample, dsize=(width, height),
                                interpolation=interpolation)
    return image_resample


if __name__ == '__main__':
    image_original = cv2.imread('')
    image_resample = resample(image_original, 2, cv2.INTER_CUBIC)

    # psnr
    psnr = compute_psnr(image_original, image_resample, 255)
    psnr_skimage = metrics.peak_signal_noise_ratio(image_original,
                                                   image_resample,
                                                   data_range=255)
    print("==============psnr===============")
    print(psnr)
    print(psnr_skimage)

    # ssim
    image_original = color_to_gray(image_original, normalization=False)
    image_resample = color_to_gray(image_resample, normalization=False)

    mssim, ssim_map = compute_ssim(image_original,
                                   image_resample)
    # set multichannel as False if using gray image, else True
    mssim_skimage = metrics.structural_similarity(image_original,
                                                  image_resample,
                                                  multichannel=False)
    print(mssim)
    print(mssim_skimage)

    # cv2.imshow('original', image_original)
    # cv2.imshow('resample', image_resample)
    # cv2.imshow('ssim_map', ssim_map)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
