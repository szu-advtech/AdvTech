import numpy as np
import time

# 计算IoU
def cal_iou(outputs, gt):
    """IoU = intersection(A, B) / union(A, B)

    :param outputs: Network Output Mask
    :param gt:      Mask GroudTruth
    :return: IoU
    """
    eps = 1e-8
    batch_size = outputs.shape[0]
    iou = 0

    for idx in range(batch_size):
        intersection = np.logical_and(outputs[idx], gt[idx])
        union = np.logical_or(outputs[idx], gt[idx])
        intersection = np.sum(intersection)
        union = np.sum(union)
        iou += intersection / (union + eps)
    iou = iou / batch_size
    return iou
