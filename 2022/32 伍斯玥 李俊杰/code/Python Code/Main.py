#coding=utf-8
import Guide
import Graphstructure as G
import Comparedalgorithm as C

# 决定抽样比例
rate=0.1

# 读入被抽样图
graph = G.getgraph("data/Haggle.csv")
print(graph)
sample_nodes_num=G.get_number_of_nodes(graph)*rate

# 抽样子图
# ForestFire
s_FF=C.forestfire(graph,sample_nodes_num)
print(s_FF)
# Random Walk
s_RW=C.random_walk_sampling_simple(graph,sample_nodes_num)
print(s_RW)
# Snowball
s_SB=C.snowball(graph,sample_nodes_num,8)
print(s_SB)

# Guided Sampling
# c与d可以自行输入自己想要的值
c=G.AVGclusteringcoefficient(graph)
print(c)
d=G.AVGdegree(graph)
s_Guide=Guide.Guidesampling(graph,d,c,rate=rate)
print(s_Guide)