#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python training_CUB.py --loc_model resnet50 --pseudo_bboxes_path /home/wxq/workspace/Latest-C2AM/CCAM-master/WSOL/debug/images/CCAM_CUB_IP/train/pseudo_boxes/ --save_path CUB_resnet50_uns /home/wxq/mydataset/CUB_200_2011/
# CUDA_VISIBLE_DEVICES=4 python inference_CUB.py --loc_model resnet50 --cls_model efficientnetb7 --ckpt ./CUB_resnet50_uns/checkpoint_localization_cub_resnet50_99.pth.tar --cls_ckpt ./efficientnetb7-cub200-best-epoch.pth /path/to/CUB/
CUDA_VISIBLE_DEVICES=0 python inference_CUB.py --loc_model resnet50 --cls_model efficientnetb7 --ckpt ./CUB_resnet50_uns/checkpoint_localization_cub_resnet50_99.pth.tar --cls_ckpt ./efficientnetb7-cub200-best-epoch.pth /home/wxq/mydataset/CUB_200_2011/
