#!/usr/bin/env bash

set -x
NGPUS=$1
PY_ARGS=${@:2}

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=${NGPUS} train.py --launcher pytorch ${PY_ARGS}
