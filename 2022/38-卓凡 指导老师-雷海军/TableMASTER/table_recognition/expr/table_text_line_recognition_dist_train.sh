# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=29500 ./tools/dist_train.sh ./configs/textrecog/master/master_lmdb_ResnetExtra_tableRec_dataset_dynamic_mmfp16.py ./work_dir/1123_MASTER_textline/ 8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 PORT=29500 ./tools/dist_train.sh ./configs/textrecog/master/master_lmdb_ResnetExtra_tableRec_dataset_dynamic_mmfp16.py ./work_dir/1123_MASTER_textline/ 7
