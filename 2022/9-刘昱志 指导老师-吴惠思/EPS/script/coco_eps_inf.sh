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

CUDA_VISIBLE_DEVICES=${GPU} python3 infer.py \
    --dataset coco \
    --infer_list metadata/coco/train.txt \
    --img_root /data/liuyuzhi/EPS/coco14/train2014 \
    --network network.resnet38_eps \
    --weights /home/liuyuzhi/code/EPS/save_info/coco_eps/checkpoint_cls.pth \
    --thr 0.20 \
    --n_gpus 4 \
    --n_processes_per_gpu 1 1 1 1\
    --cam_png /home/liuyuzhi/code/EPS/save_info/coco_eps/result/cam_png