## Environment

- PyTorch 1.1.0
- python3+

## Setup

```bash
cd ext/nms && make && cd ..

# Prepare Drones dataset in 'data' folder.

# For Training RRNet
python3 train.py

# For evaluation
python3 eval.py
```