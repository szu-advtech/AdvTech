#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=6

# conda activate echog
# python -m torch.distributed.launch --nproc_per_node=NUM_GPUS_YOU_HAVE main.py
# nproc_per_node: 这个参数是指你使用这台服务器上面的几张显卡


##########################################################################################################
# one frame
# one gpu
# python train.py --config_file files/configs/Train_single_frame.yaml TRAIN.BATCH_SIZE 128
nohup python train.py --config_file files/configs/Train_single_frame.yaml > /dev/null &

# nohup python train.py --config_file files/configs/Train_single_frame.yaml TRAIN.BATCH_SIZE 128 > /dev/null &

# multiple GPUs
# python -m torch.distributed.launch --nproc_per_node=6 train.py --config_file files/configs/Train_single_frame.yaml TRAIN.BATCH_SIZE 256

##########################################################################################################
# multi frame
# python train.py --config_file files/configs/Train_multi_frame.yaml 
# python train.py --config_file files/configs/Train_multi_frame_SD.yaml 

# nohup python train.py --config_file files/configs/Train_multi_frame_SD.yaml > /dev/null &
# nohup python train.py --config_file files/configs/Train_multi_frame_SD.yaml > /dev/null &
 

# nohup python train.py --config_file files/configs/Train_multi_frame_SD.yaml TRAIN.BATCH_SIZE 16 > /dev/null &

