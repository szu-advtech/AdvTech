# mipnerf_pl

## Installation

```

conda create --name mipnerf python=3.9
conda activate mipnerf
pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
```
## Dataset
```
python datasets/convert_blender_data.py --blender_dir UZIP_DATA_DIR --out_dir OUT_DATA_DIR

```
## Train
To train a single-scale `lego` Mip-NeRF:
```
python train.py --out_dir OUT_DIR --data_path UZIP_DATA_DIR --dataset_name blender exp_name EXP_NAME
```
To train a multi-scale `lego` Mip-NeRF:
```
python train.py --out_dir OUT_DIR --data_path OUT_DATA_DIR --dataset_name multi_blender exp_name EXP_NAME
```

## Evaluation

```
python eval.py --ckpt CKPT_PATH --out_dir OUT_DIR --scale 4 --save_image

python eval.py --ckpt CKPT_PATH --out_dir OUT_DIR --scale 4 --summa_only
```

