# NSGANetV1
本代码实现基于论文
> Multiobjective Evolutionary Design of Deep Convolutional Neural Networks for Image Classification <br>
> Zhichao Lu, Ian Whalen, Yashesh Dhebar, Kalyanmoy Deb, Erik D. Goodman, Wolfgang Banzhaf, and Vishnu Naresh Boddeti

# 环境需求
> Python >= 3.9.13, Pytorch >= 1.7.1, torchvision >= 0.8.2, pymoo == 0.3.0

# 搜索架构
运行搜索架构程序指令
> python search.py --pop_size 15 --n_gens 10 --n_offspring 10 --init_channels 24 --layers 11 --epochs 30 --device cuda:0

结果保存于 result 文件夹中的 fitness 和 individuals 文件。

# 训练架构
在完成搜索后，进行架构训练,假设搜索结果路径为 ./result/individuals
> python train.py --genofile ./result/individuals --n_geno 1 --epochs 600 --device cuda:0

其中，n_geno 为搜索结果中的个体序号。
