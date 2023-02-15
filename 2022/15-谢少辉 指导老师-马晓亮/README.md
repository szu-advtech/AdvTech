## Create anaconda environment on Linux
```bash
conda create -n sacall python=3.7
conda activate sacall
pip install --extra-index-url https://download.pytorch.org/whl/cu113 ont-bonito==0.6.1 torchaudio torchvision tensorboard 
install apex from https://www.github.com/nvidia/apex
```

## Training dataset download
```bash
bonito download --training # training dataset is in ~/anaconda3/envs/sacall/lib/python3.7/site-packages/bonito/data/ 
```

### Traning command
```bash
### train sacall model
python transformer_train.py --epochs 30 --batch-size 64 --eval-batch-size 64 --lr-warmup-ratio 0.02 --lr-hold-ratio 0.98 --lr-decay-ratio 0.0 --lr 0.0001 --lr-end 1e-5 --print-freq 10 --eval-freq 1 --ngpus-per-node 2 [bonito_training_dataset_dir] [training_output_dir]

### train sacall-conv model
python transformer_train.py --epochs 30 --batch-size 64 --eval-batch-size 64 --use-conv-transformer-encoder --lr-warmup-ratio 0.02 --lr-hold-ratio 0.98 --lr-decay-ratio 0.0 --lr 0.0001 --lr-end 1e-5 --print-freq 10 --eval-freq 1 --ngpus-per-node 2 [bonito_training_dataset_dir] [training_output_dir]
```

### Basecalling command
Note that before basecalling, you need to decompress the model files in the model folder with 7z. 
```bash
python fast5_preprocess.py --chunksize 3600 --overlap 400 --nproc 1 fast5_sample fast5_preprocess_output

python transformer_basecaller.py --model model/SACall_model.pth.tar --batch-size 1 --gpu 0 --seed 40 --chunksize 3600 --overlap 400 fast5_preprocess_output sacall_model_output 

python transformer_basecaller.py --model model/SACall_Conv_model.pth.tar --use-conv-transformer-encoder --batch-size 1 --gpu 0 --seed 40 --chunksize 3600 --overlap 400 fast5_preprocess_output sacall_conv_model_output
```
