import Graphstructure as G
import ModDFS
import networkx as nx
import numpy as np

def weight(edge_init,graph):
    # 初始化权重列表
    weight=list()
    edge_init=list(edge_init)
    # 为需要删除的每条边计算权重,权重其实是该边对全图聚类系数的影响程度
    for i in range(len(edge_init)):
        # 获得能与该边形成三角形的节点数目，该边通过影响这些节点的聚类系数从而影响全图聚类系数
        tempedge=list(edge_init[i])
        n1 = tempedge[0]
        n2 = tempedge[1]
        nn1=G.get_neighbors(graph, n1)
        nn2=G.get_neighbors(graph, n2)
        if n1!=n2:# 去掉自环，自环定义为w=0
            n=list(set(nn1).intersection(set(nn2)))
        else:
            n=list()
        # 计算该边的权重
        w = 0
        if len(n)>0:
            for j in range(len(n)):
                k=G.get_degree(graph,n[j])
                w=w+(2/(k*(k-1)))
                weight.append(w)
        else:
            weight.append(w)
    # 合并边与对应权重
    z=list()
    for i in range(len(edge_init)):
        tempedge=list(edge_init[i])
        tempedge.append(weight[i])
        z.append(tempedge)
    # 将边按照权重升序排列
    weight_edge=sorted(z, key=lambda tup: tup[2], reverse=True)
    return weight_edge

def Guidesampling(graph,d_org,c_org,rate=0.01,seed=42,start_node=None):
    # 为原图进行ModDFS抽样后再诱导所有边
    ModDFS_induce=ModDFS.ModDFS_induce(graph,start_node=start_node,rate=rate,seed=seed)
    sampler=nx.Graph(ModDFS_induce)
    VS_NUM=G.get_number_of_nodes(sampler)
    ES_NUM=G.get_number_of_edges(sampler)
    # ModDFS样本图的边
    edge_init = G.get_edges(sampler)
    # 为ModDFS样本图的边，基于原图计算权重（可以每条边都计算，为了减少运算量，只计算需要删除的边）
    edge_weight=weight(edge_init,graph)
    c = 2
    edge_init = []
    for l in edge_weight:
        # 删除一行中第c列的值
        rest_l = l[:c]
        rest_l.extend(l[c + 1:])
        # 将删除后的结果加入结果数组
        edge_init.append(rest_l)
    # 不能删除权值为0的边，以免造成孤立点
    zero_weight = 0
    for i in range(len(edge_weight)):
        if(edge_weight[i][2]==0):
            zero_weight=zero_weight+1
    # 计算需要删除的边数目
    e_extra=round(ES_NUM-d_org*VS_NUM/2)
    # 此时的聚类系数
    c_init=G.AVGclusteringcoefficient(sampler)
    # 已经删除的节点数
    e_del=0
    # 剩下的还需要删除边的占比
    e_ratio=1
    #目前的聚类系数值
    c_curr=c_init
    # 目前聚类系数完成目标比例
    if c_curr != 0:
        c_ratio=c_org/c_curr
    else:
        c_ratio=0
    # 利用斜率判断需要需要删除的边的位置
    slope=(c_curr-c_org)/(c_org*e_extra)
    # 聚类系数希望在本次删边后的的期望值
    c_exp=c_init-(slope*e_del*c_org)
    while e_del<e_extra:# 只要没删够就一直删
    # 如果权重是等差那么slope是定值，且每次删边后c_exp每次都能达到，如果c_curr比c_exp表示这次选择的边权重太小，下次增大，反正同理
        if c_curr>c_exp and c_curr>c_org:
            mid = ES_NUM / 2
            index=round(mid*c_ratio*e_ratio)-1# 减1是因为索引从0开始
        else:
            ES_NUM = ES_NUM - zero_weight
            mid = ES_NUM / 2  # 高于50%为高权值，低于50%为低权值
            index=round(mid+mid*e_ratio)-1
        if index <0:
            index=0
        edge_init.remove(edge_init[index])
        sampler=nx.from_edgelist(edge_init)
        e_del=e_del+1
        c_curr=G.AVGclusteringcoefficient(sampler)
        if c_curr!=0:
            c_ratio=c_org/c_curr
        else:
            c_ratio = 0
        e_ratio=(e_extra-e_del)/e_extra
        c_exp=c_init-(slope*e_del*c_org)
    return sampler