import numpy as np
import os
from chainercv.datasets import VOCSemanticSegmentationDataset
from chainercv.evaluations import calc_semantic_segmentation_confusion

from misc import imutils


def run(args):
    dataset = VOCSemanticSegmentationDataset(split=args.chainer_eval_set, data_dir=args.voc12_root)

    preds = []
    labels = []
    n_images = 0
    id_list = []
    for file in os.listdir("/data2/xiepuxuan/code/AMR/outputs/amr_voc2012_2/cam_outputs/"):
        id_list.append(file[:-4])
    dataset.ids = id_list
    for i, id in enumerate(dataset.ids):
        n_images += 1
        cam_dict = np.load(os.path.join(args.cam_out_dir, id + '.npy'), allow_pickle=True).item()
        cams = cam_dict['high_res']

        cams = np.expand_dims(cams, axis=0) if (cams.ndim < 3) else cams

        cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=args.cam_eval_thres)
        keys = np.pad(cam_dict['keys'] + 1, (1, 0), mode='constant')

        cls_labels = np.argmax(cams, axis=0)

        cls_labels = keys[cls_labels]
        preds.append(cls_labels.copy())
        labels.append(dataset.get_example_by_keys(i, (1,))[0])

    confusion = calc_semantic_segmentation_confusion(preds, labels)

    gtj = confusion.sum(axis=1)
    resj = confusion.sum(axis=0)
    gtjresj = np.diag(confusion)
    denominator = gtj + resj - gtjresj
    iou = gtjresj / denominator

    print("threshold:", args.cam_eval_thres, 'miou:', np.nanmean(iou), "i_imgs", n_images)
    print('among_predfg_bg', float((resj[1:].sum() - confusion[1:, 1:].sum()) / (resj[1:].sum())))
