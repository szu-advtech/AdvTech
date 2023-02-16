SCRIPT_FLAGS="--method_type mcddpm \
--log_dir logs/fastmri/mcddpm/pd4x"
DATASET_FLAGS="--dataset fastmri --data_dir ../datasets/fastmri/knee_singlecoil_train \
--data_info_list_path data/fastmri/pd_train_info.pkl \
--batch_size 16 --acceleration 4 --num_workers 6"
TRAIN_FLAGS="--microbatch 4 --log_interval 10 --save_interval 5000 --max_step 35000 \
--model_save_dir checkpoints/fastmri/mcddpm/pd4x"

python train.py $SCRIPT_FLAGS $DATASET_FLAGS $TRAIN_FLAGS