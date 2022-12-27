==================================================================
For datasets
CUB-200-2011
you will need to download the image and make sure your data/CUB_200_2011folder is structured as follows:
|--CUB_200_2011/
|------images
|------images.txt
|------bounding_boxes.txt
|------......
|------train_test_split.txt

PASCAL VOC2012
you will need to download images and make sure your data/VOC2012 folder is structured as follows:
|── VOC2012/
|------Annotations
|------ImageSets
|------SegmentationClass
|------SegmentationClassAug
|------SegmentationObject

==================================================================
For WSOL task
1. Train CCNN on CUB-200-2011 dataset
cd WSOL
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1 python train_CCNN_CUB.py --experiment CCNN_CUB_MOCO --lr 0.0001 --batch_size 16 --pretrained mocov2 --alpha 0.75
2. Train a regressor using the psuedo bboxes extracted from CCNN
cd PSOL
python3 train_test_split.py --data_dir path/to/your/dataset
CUDA_VISIBLE_DEVICES=1 python training_CUB.py --loc_model resnet50 --pseudo_bboxes_path ../debug/images/CCNN_CUB_MOCO/train/pseudo_boxes/ --save_path CUB_resnet50_uns path/to/your/dataset
CUDA_VISIBLE_DEVICES=1 python inference_CUB.py --loc_model resnet50 --cls_model efficientnetb7 --ckpt ./CUB_resnet50_uns/checkpoint_localization_cub_resnet50_99.pth.tar --cls_ckpt ./efficientnetb7-cub200-best-epoch.pth path/to/your/dataset

==================================================================
For WSSS task
cd WSSS
1. Train CCNN on PASCAL VOC2012 dataset
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=3 python train_CCNN_VOC12.py --tag CCNN_VOC12_MOCO --batch_size 128 --pretrained mocov2 --alpha 0.25
2. To extract class-agnostic activation maps
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=0 python inference_CCNN.py --tag CCNN_VOC12_MOCO --domain train
3. To extract background cues
OMP_NUM_THREADS=16 CUDA_VISIBLE_DEVICES=1 python inference_crf.py --experiment_name CCNN_VOC12_MOCO@train@scale=0.5,1.0,1.5,2.0 --threshold 0.3 --crf_iteration 10 --domain train
4. CAMs Refinement
python evaluate_using_background_cues.py --experiment_name CCNN_VOC12_MOCO@train@scale=0.5,1.0,1.5,2.0  --domain train --with_bg_cues True --bg_dir ../experiments/predictions/CCNN_VOC12_MOCO@train@scale=0.5,1.0,1.5,2.0@t=0.3@ccam_inference_crf=10/ --gt_dir path/to/your/dataset/VOC2012/SegmentationClass/