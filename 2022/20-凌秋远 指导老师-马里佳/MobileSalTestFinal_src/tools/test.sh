#!/bin/bash
PREFIX=./pretrained/
MODEL_NAME=mobilenet_v3LARGE__ms2_44__0.9097
MODEL_PATH=$PREFIX$MODEL_NAME.pth

python tools/test.py --pretrained $MODEL_PATH \
                                      --savedir ./maps/$MODEL_NAME/ \
                                      --depth 1



