#!/bin/bash

# custom config
# DATA=/path/to/datasets
DATA=/public/datasets
TRAINER=ZeroshotCLIP
DATASET=$1
CFG=vit_b16

SHOTS=16
SUB=new

COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}


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