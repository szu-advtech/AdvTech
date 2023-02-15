# from sklearn.metrics import f1_score
import numpy as np

def f1_score(y_true, y_pred):
    e = 1e-8
    gp = np.sum(y_true)  # 预测值0 1 TP+FP 将预测为1的加起来
    tp = np.sum(y_true*y_pred)  # 预测值和真实值相乘，只有1*1有效 得到TP值
    pp = np.sum(y_pred)    # 真实值中为1的总数 TP+FN
    p = tp/(gp+e)
    r = tp/(pp+e)
    f1 = (2 * p * r) / (p + r + e)
    return f1

def cal_f1(outputs, gt):
    """
    F1 = 2 * (precision * recall) / (precision + recall)
    precision = true_positives / (true_positives + false_positives)
    recall    = true_positives / (true_positives + false_negatives)

    :param outputs: Network Output Mask
    :param gt:      Mask GroudTruth
    :return:
    """
    batch_size = outputs.shape[0]
    f1 = 0
    for idx in range(batch_size):
        trans_output = outputs[idx].flatten()
        trans_gt = gt[idx].flatten()
        f1 += f1_score(trans_output, trans_gt)
    f1 = f1 / batch_size

    return f1


if __name__ == '__main__':
    # outputs = np.random.randint(0, 2, (1, 10))
    # print(outputs)
    # gt = np.random.randint(0, 2, (1, 10))
    # print(gt)
    outputs = np.array([[0, 1, 0, 1, 0, 1, 1, 0, 1, 0]], dtype=np.int32)
    gt = np.array([[0, 0, 1, 1, 0, 1, 1, 1, 1, 0]], dtype=np.int32)
    outputs = outputs.reshape((1, 2, 5))
    gt = gt.reshape((1, 2, 5))
    print(outputs.shape, gt.shape)
    print(cal_f1(outputs, gt))
