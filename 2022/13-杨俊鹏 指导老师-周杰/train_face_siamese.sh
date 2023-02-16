#!/usr/bin/env sh
mkdir -p log
now=$(date +"%Y%m%d_%H%M%S")
log_name="Face_Kinect_Siamese_4SAless1FC_noOC_id_LOG_"$now""
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1,2
nohup python -u train_face_siamese.py \
--config cfgs/config_ssn_face_id_siamese.yaml \
2>&1|tee log/$log_name.log &
