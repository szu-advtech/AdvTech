#!/usr/bin/env sh
now=$(date +"%Y%m%d_%H%M%S")
log_name="Kinect_4SALayer_Siamese_Evaluate_Face_ID_LOG_"$now""
export CUDA_VISIBLE_DEVICES=3,0
python -u evaluate_face_id.py \
--config cfgs/config_ssn_face_id_siamese.yaml \
2>&1|tee log/evaluate_log/$log_name.log
