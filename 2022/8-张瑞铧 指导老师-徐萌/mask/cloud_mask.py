import numpy as np
import scipy


def Rescale(band, limits):
    return (band - limits[0]) / (limits[1] - limits[0])


def get_normalized_difference(channel1, channel2):
    subchan = channel1 - channel2
    sumchan = channel1 + channel2
    sumchan[sumchan == 0] = 0.001  # checking for 0 divisions
    return subchan / sumchan


def Cloud_mask(img, T_cl=0.2, binarize=True):
    img = img / 10000.
    channels, length, width = img.shape
    cloud_score = np.ones((length, width)).astype('float32')

    # 计算cloud score
    # https://gisgeography.com/sentinel-2-bands-combinations/#Sentinel_2_Bands
    # Band1：Coastal aerosol（60m）
    # Band2：Blue（10m）
    # Band3：Green（10m）
    # Band4：Red（10m）
    # Band10: cirrus(卷云)（60m）
    # Band11：SWIR
    # NDSI：归一化积雪指数 = (GREEN - SWIR) / (GREEN + SWIR) = Band3 and Band11  用于区分云与雪
    B1 = img[0]
    B2 = img[1]
    B3 = img[2]
    B4 = img[3]
    B10 = img[9]
    B11 = img[10]
    B12 = img[11]

    cloud_score = np.minimum(cloud_score, Rescale(B2, [0.1, 0.5]))
    cloud_score = np.minimum(cloud_score, Rescale(B1, [0.1, 0.3]))
    cloud_score = np.minimum(cloud_score, Rescale(B1 + B11, [0.15, 0.2]))
    cloud_score = np.minimum(cloud_score, Rescale(B4 + B3 + B2, [0.2, 0.8]))

    """
    ndmi = (img[7] - img[11]) / (img[7] + img[11])
    cloud_score = np.minimum(cloud_score, Rescale(ndmi, [-0.1, 0.1]))
    """

    ndsi = get_normalized_difference(B3, B11)
    cloud_score = np.minimum(cloud_score, Rescale(ndsi, [0.8, 0.6]))

    # closing
    cloud_score = scipy.ndimage.morphology.grey_closing(cloud_score, size=(5, 5))

    # Average
    kernel = np.ones((7, 7)) / (7 ** 2)
    cloud_score = scipy.signal.convolve2d(cloud_score, kernel, mode='same')

    # value clipping
    cloud_score = np.clip(cloud_score, 0.00001, 1.0)

    if binarize:
        # binary threshold
        cloud_score[cloud_score >= T_cl] = 1
        cloud_score[cloud_score < T_cl] = 0

    return cloud_score


def Shadow_mask(img, T_csi=3/4, T_wbi=5/6):
    img = img / 10000.
    channels, length, width = img.shape

    B1 = img[1]
    B8 = img[7]
    B11 = img[11]
    CSI = (B8 + B11) / 2.
    WBI = B1

    shadow_mask = np.zeros((length, width)).astype('float32')

    T_csi = np.min(CSI) + T_csi * (np.mean(CSI) - np.min(CSI))
    T_wbi = np.min(B1) + T_wbi * (np.mean(B1) - np.min(B1))

    shadow_tf = np.logical_and(CSI < T_csi, WBI < T_wbi)
    shadow_mask[shadow_tf] = -1
    shadow_mask = scipy.signal.medfilt2d(shadow_mask, 5)

    return shadow_mask


def Mask_Merging(origin_img):
    cloud_img = Cloud_mask(origin_img)
    shadow_img = Shadow_mask(origin_img)

    cloud_shadow_mask = np.zeros_like(cloud_img)
    cloud_shadow_mask[shadow_img < 0] = -1
    cloud_shadow_mask[cloud_img > 0] = 1

    return cloud_shadow_mask
