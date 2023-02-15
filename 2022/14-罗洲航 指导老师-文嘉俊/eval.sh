#!/usr/bin/env bash
DTU_TESTING="/home/zhouhang/MVSNet_pytorch-master/DTU_TESTING/dtu"
CKPT_FILE="./checkpoints/train/C1_2_PD/model_000015.ckpt"
pre_CKPT_FILE="./checkpoints/search/C1_2_PD/model_000000.ckpt"
python eval.py --dataset=dtu_yao_eval --batch_size=1 --testpath=$DTU_TESTING --testlist lists/dtu/test.txt --loadckpt $CKPT_FILE --pre_loadckpt $pre_CKPT_FILE $@
