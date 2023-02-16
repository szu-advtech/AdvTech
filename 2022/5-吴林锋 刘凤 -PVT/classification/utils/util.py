import sys
import jittor as jt
import jittor.nn as nn
from tqdm import tqdm
import numpy as np


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def execute(self, x, target):
        logprobs = nn.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target)
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def train_one_epoch(model, optimizer, data_loader, epoch):
    model.train()
    loss_function = LabelSmoothingCrossEntropy(smoothing=0.1)
    # loss_function = jt.nn.CrossEntropyLoss()
    accu_loss = jt.zeros(1)  # 累计损失
    accu_num = jt.zeros(1)   # 累计预测正确的样本数
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.float32()
        sample_num += images.shape[0]
        labels = labels.unsqueeze(1)
        pred = model(images)
        pred_classes = jt.array(np.argmax(pred.data, axis=1)).unsqueeze(1)  # 按每行求出最大值的索引
        accu_num += jt.equal(pred_classes, labels).sum()

        loss = loss_function(pred, labels)
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not jt.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step(loss)
        # optimizer.zero_grad()
        # optimizer.backward(loss)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@jt.no_grad()
def evaluate(model, data_loader, epoch):
    loss_function = jt.nn.CrossEntropyLoss()
    model.eval()

    accu_num = jt.zeros(1)   # 累计预测正确的样本数
    accu_loss = jt.zeros(1)  # 累计损失
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        images = images.float32()
        sample_num += images.shape[0]
        labels = labels.unsqueeze(1)
        pred = model(images)
        pred_classes = jt.array(np.argmax(pred.data, axis=1)).unsqueeze(1)  # 按每行求出最大值的索引
        accu_num += jt.equal(pred_classes, labels).sum()
        loss = loss_function(pred, labels)
        accu_loss += loss.detach()

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)
    return accu_loss.item() / (step + 1), accu_num.item() / sample_num
