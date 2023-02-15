## RSTPM

原文连接：https://arxiv.org/abs/2210.07548

## 整体架构图

![overview](https://github.com/todofirst/AdvTech/blob/main/2022/11-%E9%99%88%E4%BF%AE%E5%BB%BA%20%E6%8C%87%E5%AF%BC%E8%80%81%E5%B8%88-%E9%AB%98%E7%81%BF/overview.png "overview")

## 训练与测试

两对student-teacher和判别网络是分开训练，数据库基于MvTec AD数据库，整体实现基于pytorch-lightning



两对student-teacher训练

```shell
python RSTPMtrain.py --phase train --num_epochs 100 --dataset_path /data/cxj/datasets/mvt   --category all --project_path /data/cxj/results/RSTPM15
```

- phase：表示当前是训练(train)还是测试(test)阶段
- category：表示训练的类别，all表示所有15个类
- project_path：表示保存的项目数据的地址

判别网络训练

```shell
python DisTrain.py --phase train --num_epochs 300 --dataset_path /data/cxj/datasets/mvt --anomaly_source_path /data/cxj/datasets/dtd/images  --category toothbrush --project_path /data/cxj/results/Dis3 --stpm_ckpt_path /data/cxj/results/RSTPM12
```

- anomaly_sorce_path为异常源图片
- stpm_ckpt_path：为上述训练好的两对student-teacher模型



