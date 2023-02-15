# 环境
本人实验环境：系统：ubuntu20.04.5LTS

GPU：A3000

cuda版本：11.3 

安装环境：

1，创建conda虚拟环境:conda create -n nerf-w python=3.8

2，进入环境：conda activate nerf-w

3，安装torch：pip install torch==1.11.0 --extra-index-url 
https://download.pytorch.org/whl/cu113

4，安装torch-scatter：pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+${CUDA}.html

5，安装tinycudann:git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
                 cd tiny-cuda-nn
                 cmake . -B build
                 cmake --build build --config RelWithDebInfo -j
                 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
                 cd bindings/torch
                 python setup.py install
6，安装apex：pip install apex

4，安装包：pip install pip install -r requirements.txt

# 运行

python train.py 
   --dataset_name blender 
   --root_dir $BLENDER_DIR 
   --N_importance 64 --img_wh 400 400 --noise_std 0 
   --num_epochs 20 --batch_size 1024 
   --optimizer adam --lr 5e-4 --lr_scheduler cosine 
   --exp_name exp 
   --data_perturb color occ 
   --encode_t 
   --encode_a
   --beta_min 0.1
