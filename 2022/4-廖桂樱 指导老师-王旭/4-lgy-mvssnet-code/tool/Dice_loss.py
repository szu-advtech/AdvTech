import torch
import torch.nn as nn
import numpy as np

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def forward(output, target, smooth=1e-5):

        batch = target.size(0)
        input_flat = output.view(batch, -1)
        target_flat = target.view(batch, -1)
        # print("input_flat", input_flat.shape)
        # print("target_flat", target_flat.shape)
        intersection = input_flat * target_flat
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / batch
        return loss

if __name__ == '__main__':
    a = np.array([[[1,2],[3,4]],
         [[5,6],[7,8]],[[9,10],[11,12]]])
    a = torch.tensor(a)
    # outputs = np.array([[[0, 5, 0, 2, 0, 1, 1, 0, 1, 0], [0, 1, 0, 2, 0, 1, 1, 0, 1, 0]]], dtype=np.int32)
    # outputs = torch.tensor(outputs)
    print(a.shape)
    print(a.size(0))