import jittor as jt
import jittor.nn as nn


def dice_coeff(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.shape == target.shape
    if input.ndim == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')
    if input.ndim == 2 or reduce_batch_first:
        # inter = jt.dot(input.reshape(-1), target.reshape(-1))
        inter = input.reshape(1,-1) @ target.reshape(-1,1)
        sets_sum = input.sum() + target.sum()
        if sets_sum.item() == 0:
            sets_sum = 2 * inter
        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input, target, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.shape == target.shape
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]


def dice_loss(input, target, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.shape == target.shape
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


class DiceLoss(nn.Module):
    def __init__(self, smooth=1):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def execute(self, input, target):
        '''
        :param input:  [batch_size,c,h,w]
        :param target: [batch_size,h,w]
        :return: loss
        '''
        input = nn.softmax(input, dim=1)

        c_dim = input.shape[1]
        input = input.transpose((0, 2, 3, 1)).reshape((-1, c_dim))

        target = target.reshape((-1,))
        target = target.broadcast(input, [1])
        target = target.index(1) == target

        smooth = 1.0
        iflat = input.reshape((- 1)).float()
        tflat = target.reshape((- 1)).float()
        intersection = (iflat * tflat).sum()
        return (1 - (((2.0 * intersection) + smooth) / ((iflat.sum() + tflat.sum()) + smooth)))