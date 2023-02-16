GPU=0,1,2,3
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


# 1. train classification network with EPS
CUDA_VISIBLE_DEVICES=0,1 python3 main.py \
    --dataset coco \
    --train_list metadata/coco/train.txt \
    --session coco_eps \
    --network network.resnet38_eps \
    --data_root /data/liuyuzhi/EPS/coco14/train2014 \
    --saliency_root /data/liuyuzhi/EPS/coco14/SALImages \
    --weights /data/liuyuzhi/EPS/coco14/ilsvrc-cls_rna-a1_cls1000_ep-0001.params \
    --resize_size 256 448 \
    --crop_size 321 \
    --tau 0.4 \
    --lam 0.9 \
    --max_iters 256500 \
    --batch_size 16