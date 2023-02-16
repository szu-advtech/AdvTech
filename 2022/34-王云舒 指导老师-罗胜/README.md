<div align="center">
  <h1>原型网络复现工作</h1>
</div>

<div align="center">
  <h3>王云舒 2200271052</h3>
</div>


## :heavy_check_mark: Requirements
* Ubuntu 20.04
* Python 3.8
* [CUDA 11.0](https://developer.nvidia.com/cuda-toolkit)
* [PyTorch 1.7.1](https://pytorch.org)


## :gear: Conda environmnet installation
```bash
conda env create --name yourname --file environment.yml
conda activate yourname
```

## :books: Datasets
```bash
cd datasets
bash download_miniimagenet.sh
```
```
The file structure should be as follows:

    code/
    ├── datasets/
    ├── 官方原型网络/
    ├── SimpleCNAPs部分/
    ├── 复现原型网络基础与改进/
    │   ├── finetune/
    │   ├── coattention/
    │   ├── backbone conv4net&resnet12/
    │   └── t-SNE visualization/
    README.md
    environment.yml

