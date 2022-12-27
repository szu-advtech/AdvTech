# XMem

## Long-Term Video Object Segmentation with an Atkinson-Shiffrin Memory Model

### [[Failure Cases]](docs/FAILURE_CASES.md)

## Features

* Handle very long videos with limited GPU memory usage.
* Quite fast. Expect ~20 FPS even with long videos (hardware dependent).
* Come with a GUI (modified from [MiVOS](https://github.com/hkchengrex/MiVOS/tree/MiVOS-STCN)).

### Table of Contents

1. [Introduction](#introduction)
2. [Results](docs/RESULTS.md)
3. [Interactive GUI demo](docs/DEMO.md)
4. [Training/inference](#traininginference)

### Introduction

![framework](https://imgur.com/ToE2frx.jpg)

We frame Video Object Segmentation (VOS), first and foremost, as a *memory* problem.
Prior works mostly use a single type of feature memory. This can be in the form of network weights (i.e., online learning), last frame segmentation (e.g., MaskTrack), spatial hidden representation (e.g., Conv-RNN-based methods), spatial-attentional features (e.g., STM, STCN, AOT), or some sort of long-term compact features (e.g., AFB-URR).

Methods with a short memory span are not robust to changes, while those with a large memory bank are subject to a catastrophic increase in computation and GPU memory usage. Attempts at long-term attentional VOS like AFB-URR compress features eagerly as soon as they are generated, leading to a loss of feature resolution.

Our method is inspired by the Atkinson-Shiffrin human memory model, which has a *sensory memory*, a *working memory*, and a *long-term memory*. These memory stores have different temporal scales and complement each other in our memory reading mechanism. It performs well in both short-term and long-term video datasets, handling videos with more than 10,000 frames with ease.

### Training/inference

First, install the required python packages and datasets following [GETTING_STARTED.md](docs/GETTING_STARTED.md).

For training, see [TRAINING.md](docs/TRAINING.md).

For inference, see [INFERENCE.md](docs/INFERENCE.md).
