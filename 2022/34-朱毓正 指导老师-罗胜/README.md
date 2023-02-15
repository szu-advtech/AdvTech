# BPD [[arXiv](https://arXiv.org/abs/2203.14952)] | [[ACM](https://dl.acm.org/doi/abs/10.1145/3517252)]
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2202.07260)

Official pytorch implementation of "Learning Disentangled Behaviour Patterns for Wearable-based Human Activity
Recognition". (Ubicomp 2022)

Learning Disentangled Behaviour Patterns for Wearable-based Human Activity Recognition (accepted at Proceedings of the
ACM on Interactive, Mobile, Wearable and Ubiquitous Technologies, 2022) aims to solve the intra-class variability
challenge in Human activity recognition community. The proposed Behaviour Pattern Disentanglement (BPD) framework can
disentangle the behavior patterns from the irrelevant noises such as personal styles or environmental noises, etc.

对模型进行了修改，添加SEnet, ECAnet, CBAM block三个模块，并添加接口。添加经Matlab数据处理过后的PAMAP2数据；对日志功能进行修改。修改程序中的错误，将各数据集对应的维度重新处理。

由于上传GitHub文件不能太大，数据集请自行下载处理为.mat文件后放在/data/对应的数据集中。

