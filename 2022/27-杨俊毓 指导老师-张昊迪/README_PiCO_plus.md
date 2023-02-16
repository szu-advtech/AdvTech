# PiCO+: Contrastive Label Disambiguation for Robust Partial Label Learning

This is a [PyTorch](http://pytorch.org) implementation of PiCO+, a robust extention of PiCO that is able to mitigate the noisy partial label learning problem.

## Start Running PiCO+

**Run cifar10 with q=0.5 eta=0.2**

```shell
CUDA_VISIBLE_DEVICES=0 python -u train_pico_plus.py \
   --exp-dir experiment/PiCO_plus-CIFAR-10-Noisy --dataset cifar10 --num-class 10\
   --dist-url 'tcp://localhost:10007' --multiprocessing-distributed --world-size 1 --rank 0 --seed 123\
   --arch resnet18 --moco_queue 8192 --prot_start 5 --lr 0.01 --wd 1e-3 --cosine --epochs 800 --print-freq 100\
   --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.5
```

**Run cifar100 with q=0.05 eta=0.2**

```shell
CUDA_VISIBLE_DEVICES=1 python -u train_pico_plus.py \
   --exp-dir experiment/PiCO-CIFAR-100-Noisy --dataset cifar100 --num-class 100\
   --dist-url 'tcp://localhost:10018' --multiprocessing-distributed --world-size 1 --rank 0 --seed 123\
   --arch resnet18 --moco_queue 8192 --prot_start 50 --lr 0.01 --wd 1e-3 --cosine --epochs 800\
   --print-freq 100 --loss_weight 0.5 --proto_m 0.99 --partial_rate 0.05 --chosen_neighbors 5
```

yjuny:
- 对比学习框架中：PLUS版本考虑当训练数据集中有噪声样本(即真实标签并不在候选标签集合里面)时，用sel_stats字典中的dist和is_rel通过函数reliable_set_selection对样本的可靠程度进行一个排序然后选择Top-num可靠的样本进行正常的训练，
然而对于不可靠的样本会通过一个ur_weight权重修改他的loss占比权重。
- 原型/伪标签的更新：会先对噪声样本进行标签的预测，然后再用原来的方法进行原型向量和伪标签的更新