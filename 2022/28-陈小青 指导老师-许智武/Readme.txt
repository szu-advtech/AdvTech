数据集网址：https://grouplens.org/datasets/movielens/1m/

rib.py是基于PyTorch实现的RIB；

rib2.py是在rib.py的基础上将单个GRU的输出值替换为2个GRU的输出平均值；

rib3.py是在rib.py的基础上将单个GRU的输出值替换为2个MGRU的输出平均值；

main3.py是运行rib3.py的文件。