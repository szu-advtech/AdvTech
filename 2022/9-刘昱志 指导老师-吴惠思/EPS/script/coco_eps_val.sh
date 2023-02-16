GPU=0,1
DATASET_ROOT=/data/liuyuzhi/EPS/coco14
WEIGHT_ROOT=/data/liuyuzhi/EPS/coco14
SALIENCY_ROOT=/data/liuyuzhi/EPS/coco14/SALImages
SAVE_ROOT=/home/liuyuzhi/code/EPS/save_info
SESSION=coco_eps

# Default setting
DATASET=coco
IMG_ROOT=/data/liuyuzhi/EPS/coco14/train2014
BACKBONE=resnet38_eps
BASE_WEIGHT=/data/liuyuzhi/EPS/coco14}/ilsvrc-cls_rna-a1_cls1000_ep-0001.params


INFER_DATA=train # train / train_aug
TRAINED_WEIGHT=/home/liuyuzhi/code/EPS/save_info/coco_eps/checkpoint_cls.pth

GT_ROOT=/data/liuyuzhi/EPS/coco14/SegmentationClass/

CUDA_VISIBLE_DEVICES=${GPU} python3 evaluate_png.py
    --dataset coco
    --datalist metadata/coco/train.txt
    --gt_dir /data/liuyuzhi/EPS/coco14/SegmentationClass/train2014
    --save_path /home/liuyuzhi/code/EPS/save_info/coco_eps/result/train.txt
    --pred_dir /home/liuyuzhi/code/EPS/save_info/coco_eps/result/cam_png