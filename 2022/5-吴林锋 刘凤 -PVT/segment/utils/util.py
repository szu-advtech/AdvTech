import sys
import jittor as jt
import jittor.nn as nn
from tqdm import tqdm
import numpy as np
from segment.utils.dice_score import DiceLoss
import cv2
from segment.model.config import CONFIGS
config = CONFIGS['PVT_Unet_small']

class loss_Function(nn.Module):
    def __init__(self):
        super(loss_Function, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

        self.dice = DiceLoss()

    def execute(self, pred, mask, num_classes):
        loss = self.cross_entropy(pred, mask).float32() + self.dice(nn.softmax(pred, dim=1).float32(),
                                                                    mask.float32())
        # loss = self.cross_entropy(pred, mask).float32() + dice_loss(nn.softmax(pred, dim=1).float32(),nn.one_hot(mask, num_classes).permute(0, 3, 1, 2).float32(),
        #                                                             multiclass=True)
        return loss


def train_one_epoch(model, optimizer, data_loader, epoch):
    model.train()
    loss_function = loss_Function()
    # loss_function = jt.nn.CrossEntropyLoss()
    accu_loss = jt.zeros(1)  # 累计损失
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, masks = data
        images = images.float32()
        sample_num += images.shape[0]
        pred = model(images)

        # loss = loss_function(pred, masks,config.num_classes)
        loss = loss_function(pred, masks, config.num_classes)
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}".format(epoch,
                                                                  accu_loss.item()/ (step + 1))

        if not jt.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)
        optimizer.step(loss)
        # optimizer.zero_grad()
        # optimizer.backward(loss)

    return accu_loss.item() / (step + 1)


@jt.no_grad()
def evaluate(model, data_loader, evaluator, best_miou, epoch):
    # loss_function = jt.nn.CrossEntropyLoss()
    loss_function = loss_Function()
    evaluator.reset()

    model.eval()
    accu_loss = jt.zeros(1)  # 累计损失
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, masks = data
        images = images.float32()
        sample_num += images.shape[0]
        pred = model(images)
        evaluator.add_batch(masks.data, np.argmax(pred.data, axis=1))
        loss = loss_function(pred, masks, config.num_classes)
        accu_loss += loss.detach()
        data_loader.desc = "[valid epoch {}] loss: {:.3f}".format(epoch,
                                                                  accu_loss.item() / (step + 1))
    Acc = evaluator.accuracy()
    Acc_class = evaluator.class_accuracy()
    mIoU = evaluator.iou()
    FWIoU = evaluator.fwiou()
    dice = evaluator.dice()
    if (mIoU > best_miou):
        best_miou = mIoU
    print(
        'Testing result of epoch {}: miou = {} Acc = {} Acc_class = {} FWIoU = {} Best Miou = {} DSC = {}'.format(epoch,
                                                                                                                  mIoU,
                                                                                                                  Acc,
                                                                                                                  Acc_class,
                                                                                                                  FWIoU,
                                                                                                                  best_miou,
                                                                                                                  dice))
    return accu_loss.item() / (step + 1),best_miou,mIoU,dice
