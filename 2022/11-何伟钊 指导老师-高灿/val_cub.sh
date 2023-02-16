GPU_ID=$1
NET=${2}
NET_SCALE=${3}
SIZE=${4}

export CUDA_VISIBLE_DEVICES=${GPU_ID}

python ./tools_cam/test_cam.py --config_file configs/CUB/${NET}_tscam_${NET_SCALE}_patch16_${SIZE}.yaml --resume conformer_best_author.pth TEST.SAVE_BOXED_IMAGE True MODEL.CAM_THR 0.1


