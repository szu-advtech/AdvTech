#!/bin/bash

# custom config
# DATA=/path/to/datasets
DATA=/public/datasets
TRAINER=ZeroshotCLIP
DATASET=$1
#CFG=$2  # rn50, rn101, vit_b32 or vit_b16
CFG=vit_b16

SHOTS=16
SUB=base

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
DIR=output/base2new/train_${SUB}/${COMMON_DIR}

if [ -d "$DIR" ]; then
    echo "Oops! The results exist at ${DIR} (so skip this job)"
else
    python train.py \
    --root ${DATA} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/ZSCLIP/${CFG}.yaml \
    --output-dir ${DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi