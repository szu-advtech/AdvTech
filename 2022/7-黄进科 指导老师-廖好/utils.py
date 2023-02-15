from typing import List
from scipy.stats import norm
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
import pandas as pd
from itertools import combinations
import pgmpy.estimators.CITests as CITests
from causallearn.search.ScoreBased.GES import ges
import sys, os
##
## return graph with type DataFrame
def getAdjmat(edges, df):
    source = edges['source']
    target = edges['target']
    columns = df.columns
    adjmat = pd.DataFrame(index=columns, columns=columns)
    adjmat[:] = False
    for i in range(0, len(source)):
        adjmat.at[source[i], target[i]] = True
    return adjmat
def narrayToEdges(graph, df):
    dict = {}
    for i in range(0, df.shape[1]):
        dict[i] = df.columns[i]
    source = []
    target = []
    for i in range(0, len(graph)):
        for j in range(0, len(graph[i])):
            if graph[i][j] == 1:
                source.append(dict[i])
                target.append(dict[j])
    return {"source": source, "target": target}
def edgeFillGraph(targetGraph, edgeSource):
    ansGraph = targetGraph.copy(True) # the changes of ansGraph will not reflect on target
    ansGraph.loc[:,:] = False #clear
    source = edgeSource['source']
    target = edgeSource['target']
    for i in range(len(source)):
        ansGraph.loc[source[i], target[i]] = True #fill
    return ansGraph
#structural hanming distance
def SHD(correct_adjmat, another_adjmat):
    #I don't use cdt.metrics.SHD, 
    #because it can't compare two DataFrames with different dimension as far as I know
    df1 = correct_adjmat.copy(True)
    df2 = another_adjmat.copy(True)
    ans = 0 # the number of add, delete, flip operations
    same = 0 # just for debug, do not return
    for i in df1.columns: #the rows is the same as cols in adjacent matrix
        for j in df1.columns:
            if i in df2.columns and j in df2.columns:
                if df1.at[i,j] != df2.at[i,j]: #need to modify df2
                    if df1.at[i,j] == False:
                        if df1.at[j,i] == True: #flip df1:j->i df2: i->j
                            df2.at[i,j] = False
                            df2.at[j,i] = True #even if df2 is bidirected, the answer is correct, just like delete a edge
                            ans+=1
                        else: #delete edge df2: i->j
                            df2.at[i,j] = False
                            ans+=1
                    else:
                        if df2.at[j,i] == True: #flip df1: i->j df2: j->i
                            df2.at[i,j] = True
                            df2.at[j,i] = False
                            ans+=1
                        else: # add edge df2: i->j
                            df2.at[i,j] = True
                            ans+=1
                else:
                    same+=1
            elif df1.at[i,j] == True: # add edge to df2
                ans+=1
    return ans
#transpose matrix
def transpose(M):
    r = len(M[0])
    ans = []
    for i in range(0, r):
        ans.append([])
    for i in range(0, len(M)):
        for j in range(0, len(M[i])):
            ans[j].append(M[i][j])
    # print("input ==== ")
    # print(M)
    # print("output ==== ")
    # print(ans)
    return ans
#shannon entrpy
def H(vec):
    ans = 0
    for i in range(0, len(vec)):
        ans += vec[i] * np.log2(vec[i])
    ans = -ans
    return ans
#entropicCausalPair:
class CausalPair(object):
    def __init__(self,data):
        self.data = data
        self.nofsamples = data.shape[0]
        self.X = data[0]
        self.Y = data[1]
        self.Xmin = np.min(data[0])
        self.Xmax = np.max(data[0])
        self.Ymin = np.min(data[1])
        self.Ymax = np.max(data[1])
def CHI_SQUARE(X, Y, Z, data):
    return CITests.chi_square(X=X, Y=Y, Z=Z, data=data, boolean=True, significance_level=0.05)
#Joint entropy minimization algorithm
def jema(M): #Joint entropy minimization algorithm which is equivalent to Minimal Entropy Coupling
    # M = transpose(M)
    e = []
    r = 1e9+10
    for i in range(0, len(M)):
        M[i].sort(reverse=True) #decreased
        r = min(r, M[i][0])
    while r > 0:
        e.append(r)
        for i in range(0, len(M)):
            M[i][0] -= r
            M[i].sort(reverse=True)
        r = 1e9+10
        for i in range(0, len(M)):
            r = min(r, M[i][0])
    return e
def Oracle_Jinke(Xi, Xj, df):
    # distr_xi = []
    # distr_xj = []
    # dictt: key is colName, values are all states
    dictt = {}
    for colName in df.columns:
        colVals = df.get(colName)
        dictt[colName] = set()
        for j in colVals:
            dictt[colName].add(j)
    distr_xj_condition_on_xi = [] # distribution: Y|X=i
    distr_Xi = [] #marginal distribution Xi
    n = df.shape[0]
    groupByXi = df.groupby(Xi)
    groupByXi_Xj = df.groupby([Xi, Xj])
    for statei in dictt[Xi]:
        count_xi = len(groupByXi.get_group(statei))
        distr_Xi.append(count_xi/n)
        probs = []
        for statej in dictt[Xj]:
            count_xi_xj = 0
            try:
                count_xi_xj = len(groupByXi_Xj.get_group((statei, statej)))
            except KeyError:
                print("", end="")
                # print("no data (%s,%s)(%d,%d)" %(Xi, Xj, statei, statej))
            probs.append(count_xi_xj/count_xi)
        distr_xj_condition_on_xi.append(probs)

    distr_xi_condition_on_xj = []
    groupByXj = df.groupby(Xj)
    distr_Xj = []
    for statej in dictt[Xj]:
        count_xj = len(groupByXj.get_group(statej))
        distr_Xj.append(count_xj/n)
        probs = []
        for statei in dictt[Xi]:
            count_xi_xj = 0
            try:
                count_xi_xj = len(groupByXi_Xj.get_group((statei, statej)))
            except KeyError:
                # print("no data (%s,%s)(%d,%d)" %(Xi, Xj, statei, statej))
                print("", end="")
            probs.append(count_xi_xj/count_xj)
        distr_xi_condition_on_xj.append(probs)

    mecYX = jema(distr_xj_condition_on_xi)
    mecXY = jema(distr_xi_condition_on_xj)
    hYX = H(mecYX)
    hXY = H(mecXY)
    hX = H(distr_Xi)
    hY = H(distr_Xj)
    # X2Y = utils.H(mec1) + utils.H(distr_Xi)# X->Y
    # Y2X = utils.H(mec2) + utils.H(distr_Xj) # Y->X
    # hYX2,hXY2,hX2,hY2 = utils.Oracle3(Xi, Xj, df, dict)
    X2Y = hYX + hX # X->Y
    Y2X = hXY + hY # Y->X
    return X2Y < Y2X # True means X->Y
# skeleton
# getSkeleton function is copied from https://github.com/Renovamen/pcalg-py
def get_neighbors(G, x: int, y: int):
    return [i for i in range(len(G)) if G[x][i] == True and i != y]
def getSkeleton(df, alpha=0.05):
    n_nodes = df.shape[1]
    def getNodesName(nodeIndex):
        nodeNames = []
        for index in nodeIndex:
            nodeNames.append(df.columns[index])
        return nodeNames
    O = [[[] for _ in range(n_nodes)] for _ in range(n_nodes)]
    # 完全无向图
    G = [[i != j for i in range(n_nodes)] for j in range(n_nodes)]
    # 点对（不包括 i -- i）
    pairs = [(i, (n_nodes - j - 1)) for i in range(n_nodes) for j in range(n_nodes - i - 1)]
    done = False
    l = 0  # 节点数为 l 的子集
    while done != True and any(G):
        done = True
        # 遍历每个相邻点对
        for x, y in pairs:
            if G[x][y] == True:
                neighbors = get_neighbors(G, x, y)  # adj(C,x) \ {y}
                if len(neighbors) >= l:  # |adj(C, x) \ {y}| > l
                    done = False
                    # |adj(C, x) \ {y}| = l
                    for K in set(combinations(neighbors, l)):
                        # 节点 x, y 是否被节点数为 l 的子集 K d-seperation
                        # 条件独立性检验，返回 p-value
                        p_value = CHI_SQUARE(df.columns[x], df.columns[y], getNodesName(K), df)
                        # p_value = gauss_ci_test2(suff_stat, x, y, list(K))
                        # 条件独立
                        if p_value >= alpha:
                            G[x][y] = G[y][x] = False  # 去掉边 x -- y
                            O[x][y] = O[y][x] = list(K)  # 把 K 加入分离集 O
                            break
        l += 1
    return np.asarray(G, dtype=int), O
#control print
def enable_print():
    sys.stdout = sys.__stdout__
def disable_print():
    sys.stdout = open(os.devnull, "w")
#ges
def ges_wrapper(df):
    disable_print()
    ges_graph = ((ges(df.values, score_func='local_score_BDeu'))['G']).graph
    enable_print()
    for i in range(ges_graph.shape[0]):
        for j in range(ges_graph.shape[1]):
            if ges_graph[i][j] == 1 and ges_graph[j][i] == -1:
                ges_graph[i][j] = 0
                ges_graph[j][i] = 1
            elif ges_graph[i][j] == -1 and ges_graph[j][i] == 1:
                ges_graph[i][j] = 1
                ges_graph[j][i] = 0
            elif ges_graph[i][j] == -1 and ges_graph[j][i] == -1:
                ges_graph[i][j] = ges_graph[j][i] = 1
            #Record[‘G’]: learned causal graph, 
            # where Record[‘G’].graph[j,i]=1 and Record[‘G’].graph[i,j]=-1 indicate i –> j; 
            # Record[‘G’].graph[i,j] = Record[‘G’].graph[j,i] = -1 indicates i — j.
    ges_edges = narrayToEdges(ges_graph, df)
    ges_adjmat = getAdjmat(ges_edges, df)
    return ges_adjmat