## Environment

- PyTorch 1.1.0
- python3+

## Setup

```bash
cd ext/nms && make && cd ..

# Prepare Drones dataset in 'data' folder.

# For Training RRNet
cp scripts/RRNet/train.py ./
python3 train.py

# For evaluation
cp scripts/RRNet/eval.py ./
python3 eval.py
```
### CUDA_VISIBLE_DEVICES=0 /home/cheny/miniconda3/envs/zgm_RRnet/bin/python  train.py