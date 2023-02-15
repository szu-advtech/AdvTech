SCRIPT_FLAGS="--method_type unet \
--log_dir logs/fastmri/unet/pd4x"
DATASET_FLAGS="--dataset fastmri --data_dir ../datasets/fastmri/knee_singlecoil_val \
--data_info_list_path data/fastmri/pd_test_6_file_info.pkl \
--acceleration 4 --num_workers 2"
TEST_FLAGS="--microbatch 10 \
--model_save_dir checkpoints/fastmri/unet/pd4x --resume_checkpoint model010000.pt \
--output_dir outputs/fastmri/unet/pd4x"

python test.py $SCRIPT_FLAGS $DATASET_FLAGS $TEST_FLAGS