#!/usr/bin/env bash
MVS_TRAINING="/home/zhouhang/MVSNet_pytorch-master/MVS_TRAINING/mvs_training/dtu"
python search.py --dataset=dtu_yao --batch_size=4 --trainpath=$MVS_TRAINING --trainlist lists/dtu/train.txt --testlist lists/dtu/test.txt --numdepth=192 --logdir ./checkpoints/search/C1_2_PD $@
