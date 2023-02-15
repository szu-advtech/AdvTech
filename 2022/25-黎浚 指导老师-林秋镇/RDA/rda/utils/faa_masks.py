import numpy as np
import torch
import matplotlib.pyplot as plt

#functions
def create_mask(ths, ref_fft):

    #null mask
    mask = torch.ones((ref_fft.shape), dtype=torch.float32)

    _, _, h, w, _ = ref_fft.shape
    b_h = np.floor((h*ths)/2.0).astype(int)
    b_w = np.floor((w*ths)/2.0).astype(int)
    if b_h == 0 and b_w ==0:
        return mask
    else:
        mask[:, :, 0:b_h, 0:b_w, :]     = 0      # top left
        mask[:, :, 0:b_h, w-b_w:w, :]   = 0      # top right
        mask[:, :, h-b_h:h, 0:b_w, :]   = 0      # bottom left
        mask[:, :, h-b_h:h, w-b_w:w, :] = 0      # bottom right

    return mask

n = 32
ths = [0.0, 1.0]



ths=list(np.arange(ths[0], ths[1]+1/n, 1/n))

fft_source = torch.zeros([1, 3, 720, 1280, 2])
fft_source_masks = []
for i in range(len(ths)-1):
    t1 = ths[i]
    t2 = ths[i+1]
    mask = create_mask(t1, fft_source) - create_mask(t2, fft_source) #(1, 3, h, w, 2)
    mask = torch.unsqueeze(mask, 1)
    fft_source_masks.append(mask)
fc_fft_source_masks = torch.cat(fft_source_masks, dim=1) #(1, n, 3, h, w, 2)

fft_target = torch.zeros([1, 3, 512, 1024, 2])
fft_target_masks = []
for i in range(len(ths)-1):
    t1 = ths[i]
    t2 = ths[i+1]
    mask = create_mask(t1, fft_target) - create_mask(t2, fft_target) #(1, 3, h, w, 2)
    mask = torch.unsqueeze(mask, 1)
    fft_target_masks.append(mask)
fc_fft_target_masks = torch.cat(fft_target_masks, dim=1) #(1, n, 3, h, w, 2)



