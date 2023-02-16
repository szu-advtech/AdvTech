import numpy as np
import torch
import argparse
import copy
import cv2

NCLASSES = 2
BATCH_SIZE = 2

#文件的加载路径
parser = argparse.ArgumentParser()
parser.add_argument('--val_txt', type=str,
                    default='D:/untitled/.idea/SS_torch/dataset/val.txt', help='about validation')
parser.add_argument('--weights', type=str,
                    default='D:/untitled/.idea/SS_torch/weights/SS_weight_3.pth', help='weights')
opt = parser.parse_args()
print(opt)

txt_path = opt.val_txt
weight = opt.weights


__all__ = ['SegmentationMetric']


class SegmentationMetric(object):  # 计算mIoU、accuracy的类
    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def pixelAccuracy(self):
        # return all class overall pixel accuracy
        # acc = (TP + TN) / (TP + TN + FP + TN)
        acc = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        acc = round(acc, 5)
        return acc

    def classPixelAccuracy(self):
        # return each category pixel accuracy(A more accurate way to call it precision)
        # acc = (TP) / TP + FP
        classAcc = np.diag(self.confusionMatrix) / \
            self.confusionMatrix.sum(axis=1)
        return classAcc

    def meanPixelAccuracy(self):
        classAcc = self.classPixelAccuracy()
        meanAcc = np.nanmean(classAcc)
        return meanAcc

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union
        mIoU = np.nanmean(IoU)
        mIoU = round(mIoU, 4)
        return mIoU

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def Frequency_Weighted_Intersection_over_Union(self):
        # FWIOU =     [(TP+FN)/(TP+FP+TN+FN)] *[TP / (TP + FP + FN)]
        freq = np.sum(self.confusion_matrix, axis=1) / \
            np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
            np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
            np.diag(self.confusion_matrix))
        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))


if __name__ == "__main__":

    sum_1 = 0  # 累加每张图片val的accuracy
    sum_2 = 0  # 累积每张图片Val的mIoU

    for i in range(len(data)):

        image = cv2.imread(
            "D:/untitled/.idea/SS_torch/dataset/jpg_right/%s" % data[i] + ".jpg", -1)
        label = cv2.imread(
            "D:/untitled/.idea/SS_torch/dataset/png_right/%s" % data[i] + ".png", -1)

        orininal_h = image.shape[0]               # 读取的图像的高
        orininal_w = image.shape[1]               # 读取的图像的宽

        image = cv2.resize(image, dsize=(416, 416))
        label = cv2.resize(label, dsize=(416, 416))

        label[label >= 0.5] = 1  # label被resize后像素值会改变,调整像素值为原来的两类
        label[label < 0.5] = 0

        image = image / 255.0          # 图像归一化
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)             # 显式的调转维度

        # 改变维度,使得符合model input size
        image = torch.unsqueeze(image, dim=0)
        image = image.type(torch.FloatTensor)             # 数据转换,否则报错
        image = image.to(device)              # 放入GPU中计算

        predict = model(image).cpu()

        # [1,1,416,416]---->[1,416,416]
        predict = torch.squeeze(predict)
        predict = predict.permute(1, 2, 0)

        predict = predict.detach().numpy()

        prc = predict.argmax(axis=-1)

        #进行mIoU和accuracy的评测
        imgPredict = prc
        imgLabel = label

        metric = SegmentationMetric(2)
        metric.addBatch(imgPredict, imgLabel)
        acc = metric.pixelAccuracy()
        sum_1 += acc
        mIoU = metric.meanIntersectionOverUnion()
        sum_2 += mIoU
        print("%s.jpg :" % data[i])
        print("accuracy:  "+str(acc*100)+" %")
        print("mIoU:  " + str(mIoU))
        print("-------------------")


    # 全部图片平均的accuracy和mIoU
    sum_1 = sum_1/len(data)
    sum_2 = sum_2/len(data)

    sum_1 = round(sum_1, 5)
    sum_2 = round(sum_2, 4)

    print("M accuracy:  "+str(sum_1*100)+" %")
    print("M mIoU:  " + str(sum_2))
