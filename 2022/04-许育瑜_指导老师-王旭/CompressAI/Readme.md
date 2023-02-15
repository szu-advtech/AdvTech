

# RGBT-LVC

### Unofficial Pytorch Implementation for

### "Learning Based Multi-Modality Image and Video Compression"

## Installation

1. Download this project.

   ```
   git clone https://github.com/xyy7/learning-RGBT-compress.git YourProjectName
   conda create -n rgbt python=3.6 
   conda activate rgbt
   pip install -r requirement.txt
   ```

2. Install compressai.

   ```
   cd YourProjectName
   pip install -U pip && pip install -e .
   ```

## Dataset

* FLIR: [download1](https://www.flir.com/oem/adas/adas-dataset-form/)  or [download2](https://pan.baidu.com/s/11GJe4MdM_NH6fuENCQ2MtQ) [password: 019b]

## Train & update

- **Pretrained model can be download through this [link](https://pan.baidu.com/s/1tUtwJufAoCeQffEpwIGgQg?pwd=pu04)**

- First, train the guided model.

  - ```
    CUDA_VISIBLE_DEVICES=1 python train.py --cuda --save -e 100 --batch-size 16 -m Guided_compresser --channel 3 -d /data/xyy/FLIR_ADAS_1_3/train/RGB --quality 5 
    ```

- Then, train the master model.

  - ```
    CUDA_VISIBLE_DEVICES=1 python train.py --cuda --save -e 100 --batch-size 8 -m Master_compresser --channel 1 -d /data/xyy/FLIR_ADAS_1_3/train/thermal_8_bit --quality 5 --checkpoint Guided_compresser_5_RGB_rgb_checkpoint_best_loss.pth.tar
    ```

## Update

```
python -m compressai.utils.update_model --architecture Guided_compresser Guided_compresser_5_RGB_rgb_checkpoint_best_loss.pth.tar -c 3

python -m compressai.utils.update_model --architecture Master_compresser Master_compresser_5_thermal_8_bit_x_checkpoint_best_loss.pth.tar -c 1
```

## Eval

```
CUDA_VISIBLE_DEVICES=1 python -m compressai.utils.eval_model.__main__rgbt checkpoint /data/xyy/FLIR_ADAS_1_3/val/thermal_8_bit20 -a  Master_compresser  -p checkpoints/Guided_compresserRGB_5_RGB_rgb_checkpoint_best_loss-*.pth.tar checkpoints/Master_compresserT_4_thermal_8_bit_x_checkpoint_best_loss-*.pth.tar -q 4 --cuda --entropy-estimation -ch 1 
```

## Codec

```
python codec_rgbt.py encode /data/xyy/FLIR_ADAS_1_3/val/thermal_8_bit20/FLIR_09063.jpeg --model Master_compresser -ch 1 -q 5 --path Guided_compresser_5_RGB_rgb_checkpoint_best_loss-*.pth.tar Master_compresser_5_thermal_8_bit_x_checkpoint_best_loss-*.pth.tar

python codec_rgbt.py decode /data/xyy/FLIR_ADAS_1_3/val/thermal_8_bit20_bin/Master_compresser_5/FLIR_09063.bin --model Master_compresser -ch 1 -q 5 --path Guided_compresser_5_RGB_rgb_checkpoint_best_loss-*.pth.tar Master_compresser_5_thermal_8_bit_x_checkpoint_best_loss-*.pth.tar
```

## BibTeX

```
@inproceedings{lu2022learning,
  title={Learning based Multi-modality Image and Video Compression},
  author={Lu, Guo and Zhong, Tianxiong and Geng, Jing and Hu, Qiang and Xu, Dong},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={6083--6092},
  year={2022}
}
```
