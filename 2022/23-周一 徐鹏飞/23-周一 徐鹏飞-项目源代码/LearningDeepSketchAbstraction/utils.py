import numpy as np
import scipy.misc
import torch
import numpy as np
from typing import List, Tuple, Union
import matplotlib
# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import PIL
import os


USE_CUDA = torch.cuda.is_available()

def to_numpy(var):
    return var.cpu().data.numpy() if USE_CUDA else var.data.numpy()

def to_tensor(ndarray, device):
    return torch.tensor(ndarray, dtype=torch.float, device=device)


def sketch_data_loading(test,dataDir):
    """Loading sketch data."""
    #classNames = ['cat', 'chair', 'face', 'firetruck', 'mosquito', 'owl', 'pig', 'purse', 'shoe']
    classNames = ['chair']
    #self.classLens = [9.80, 4.85, 6.43, 8.28, 7.18, 9.08, 9.52, 3.57, 2.89]
    numClasses = len(classNames)
    print('\n** Loading Dataset **')
    trainIterators, testIterators =  ([] for _ in range(2))
    for idx in range(numClasses):
        print('- %s' % (str(classNames[idx])))
        if (not test):
            iterator = np.load(dataDir +'train/'+ str(classNames[idx]) + '-train.npy', allow_pickle=True).item()
            iterator.raw_data, iterator.seq_len = None, None
            iterator.dataNum = len(iterator.len_strokes)
            trainIterators.append(iterator)
        iterator = np.load(dataDir +'test/'+ str(classNames[idx]) + '-test.npy', allow_pickle=True).item()
        iterator.raw_data, iterator.seq_len = None, None
        iterator.dataNum = len(iterator.len_strokes)
        testIterators.append(iterator)
    print('** Loading Dataset Complete **\n')
    return trainIterators,testIterators,numClasses


def global_to_standard(sketchBucket, sketchBucketLen):
    """Convert data with initial global coordinates to the one without initial cooridnates."""
    if len(sketchBucket) == 0: return [], [], []
    sketchArray = np.zeros([1, 3])
    sketchBucketArr = np.array(sketchBucket)
    for strokeID in range(sketchBucketArr.shape[0]):
        if strokeID == 0:
            sketchArray = np.append(sketchArray, sketchBucketArr[strokeID, 0:sketchBucketLen[strokeID] + 1, :], axis=0)
            if sum(sketchBucketArr[strokeID, 0, 0:2]) != 0:
                sketchArray[1, 0:2] = sketchArray[1, 0:2] - sketchArray[2, 0:2]
            continue
        if sum(sketchBucketArr[strokeID - 1, 0, 0:2]) == 0:
            temp = sketchBucketArr[strokeID - 1, 0:sketchBucketLen[strokeID - 1] + 1, :].sum(axis=0)
        else:
            temp = sketchBucketArr[strokeID - 1, 0:sketchBucketLen[strokeID - 1] + 1, :].sum(axis=0) - sketchBucketArr[strokeID - 1, 1, :]
        temp1 = sketchBucketArr[strokeID, 0, :] - temp  # sum of global - end point of the previous stroke
        if sum(sketchBucketArr[strokeID, 0, 0:2]) == 0: temp1 = temp1 + sketchBucketArr[strokeID, 1, :]
        tempBucketArr = sketchBucketArr[strokeID].copy()
        tempBucketArr[1, 0:2] = temp1[0:2]
        sketchArray = np.append(sketchArray, tempBucketArr[1:sketchBucketLen[strokeID] + 1, :], axis=0)
        # note: changed sketchArray[1:] to sketchArray[2:] cuz the first entry is always 3 * 0.0
    return sketchArray[2:]


# def make_image(seq: np.ndarray, path: str, name: str, show: bool, wait: Union[int, None]=None, pos: Union[Tuple[int, int], None]=None) -> None:
#     '''make_image(sequence, path, f'{name}_{label}', show=show, wait=3, pos=(10, 10))'''
#     '''Using given sequence (L, 3), draw a sketch and save it'''
#     x_ori = np.cumsum(seq[0, :, 0], axis=0)
#     print(x_ori[0:16])
#
#     y_ori = np.cumsum(seq[0, :, 1], axis=0)
#     print(y_ori[0:16])
#     z_ori = np.array(seq[0, :, 3])
#     seq = np.stack([x_ori, y_ori, z_ori]).T # 200 x 3
#
#     backend = matplotlib.get_backend()
#     print(f"backend={backend}")
#     strokes = np.split(seq, np.where(seq[:, 2] > 0)[0] + 1)
#     print(len(strokes))
#     ###
#     plt.figure()
#     '''
#     f = plt.subplots()
#     backend = matplotlib.get_backend()
#     if backend == 'TkAgg':
#         f.canvas.manager.window.wm_geometry(f'+{pos[0]}+{pos[1]}')
#     elif backend == 'WXAgg':
#         f.canvas.manager.window.SetPosition(f'+{pos[0]}+{pos[1]}')
#     else:
#         # This works for QT and GTK
#         # You can also use window.setGeometry
#         f.canvas.manager.window.move(f'+{pos[0]}+{pos[1]}')
#     '''
#     w = plt.get_current_fig_manager()
#     # if position is declared, set position
#     plt.get_current_fig_manager().window.wm_geometry(f'+{pos[0]}+{pos[1]}')
#
#     x_max, x_min = np.max(seq[:, 0]), np.min(seq[:, 0])
#     y_max, y_min = -np.min(seq[:, 1]), -np.max(seq[:, 1])
#     axis_range = max(x_max - x_min, y_max - y_min)
#     plt.xlim((x_min + x_max) / 2.0 - axis_range * 0.6, (x_min + x_max) / 2.0 + axis_range * 0.6)
#     plt.ylim((y_min + y_max) / 2.0 - axis_range * 0.6, (y_min + y_max) / 2.0 + axis_range * 0.6)
#
#     if show:
#         for s in strokes:
#             for i in range(s.shape[0] - 1):
#                 plt.pause(0.05)
#                 plt.plot(s[i:i + 2, 0], -s[i:i + 2, 1])
#         if wait is not None and wait > 0:
#             plt.pause(wait)
#     else:
#         for s in strokes:
#             plt.plot(s[:, 0], -s[:, 1])
#
#     canvas = plt.get_current_fig_manager().canvas
#     canvas.draw()
#     pil_image = PIL.Image.frombytes('RGB', canvas.get_width_height(), canvas.tostring_rgb())
#     name = f'output_{name}_strokes_{len(strokes)}.jpg'
#     pil_image.save(os.path.join(path, name), 'JPEG')
#
#     if show:
#         plt.show(block=False)
#     else:
#         plt.close()


class DataLoader(object):
	pass