# NEED TO SET
GPU=0,1
DATASET_ROOT=/data/liuyuzhi/EPS/voc12/VOCdevkit/VOC2012/
WEIGHT_ROOT=/data/liuyuzhi/EPS/coco14
SALIENCY_ROOT=/data/liuyuzhi/EPS/voc12/SALImages
SAVE_ROOT=/home/liuyuzhi/code/EPS/save_info
SESSION=voc12_eps

# Default setting
DATASET=voc12
IMG_ROOT=/data/liuyuzhi/EPS/voc12/VOCdevkit/VOC2012/JPEGImages
BACKBONE=resnet38_eps
BASE_WEIGHT=/data/liuyuzhi/EPS/coco14/ilsvrc-cls_rna-a1_cls1000_ep-0001.params


# 1. train classification network with EPS
CUDA_VISIBLE_DEVICES=${GPU} python3 main.py
    --dataset voc12
    --train_list metadata/voc12/train_aug.txt
    --session voc12_eps
    --network network.resnet38_eps
    --data_root /data/liuyuzhi/EPS/voc12/VOCdevkit/VOC2012/JPEGImages
    --saliency_root /data/liuyuzhi/EPS/voc12/SALImages
    --weights /data/liuyuzhi/EPS/coco14/ilsvrc-cls_rna-a1_cls1000_ep-0001.params
    --crop_size 448
    --tau 0.4
    --max_iters 20000
    --batch_size 8


# 2. inference CAM
INFER_DATA=train # train / train_aug
TRAINED_WEIGHT=/home/liuyuzhi/code/EPS/save_info/voc12_eps/checkpoint_cls.pth

CUDA_VISIBLE_DEVICES=${GPU} python3 infer.py
    --dataset voc12
    --infer_list metadata/voc12/train.txt
    --img_root /data/liuyuzhi/EPS/voc12/VOCdevkit/VOC2012/JPEGImages
    --network network.resnet38_eps
    --weights /home/liuyuzhi/code/EPS/save_info/voc12_eps/checkpoint_cls.pth
    --thr 0.20
    --n_gpus 4
    --n_processes_per_gpu 1 1 1 1
    --cam_png /home/liuyuzhi/code/EPS/save_info/voc12_eps/result/cam_png

# 3. evaluate CAM
GT_ROOT=/data/liuyuzhi/EPS/voc12/VOCdevkit/VOC2012/SegmentationClass/

CUDA_VISIBLE_DEVICES=${GPU} python3 evaluate_png.py
    --dataset voc12
    --datalist metadata/voc12/train.txt
    --gt_dir /data/liuyuzhi/EPS/voc12/VOCdevkit/VOC2012/SegmentationClass/
    --save_path /home/liuyuzhi/code/EPS/save_info/voc12_eps/result/train.txt
    --pred_dir /home/liuyuzhi/code/EPS/save_info/voc12_eps/result/cam_png