import utils as utils
from typing import List
import numpy as np
import bnlearn as bn
import datetime
gl_changed = {}
gl_entropy_history = {}
gl_nodes = []
gl_states = {} #key is node name, value are states
gl_group_cache = {}
def checkProp(props):
    sum = 0
    for prop in props:
        sum += prop
    return abs(sum - 1.0) < 1e-6
# calculate distribution x | S, stores in distribution_x, sVals is the concrete value of S
def dfs(x, S:List, sVals, sValsIndex, s_group, s_x_group, distribution_x):
    if sValsIndex == len(S):
        try:
            all = len(s_group.get_group(tuple(sVals) if len(sVals) > 1 else sVals[0]))
        except KeyError:
            return
        props = []
        for state in gl_states[x]:
            sVals.append(state)
            count = 0
            try:
                count = len(s_x_group.get_group(tuple(sVals) if len(sVals) > 1 else sVals[0]))
            except KeyError:
                print("", end="")
            sVals.pop()
            props.append(count/all)
        # check sum of props is equal to 1 or not
        if checkProp(props) == False:
            print("checkProp: the sum of props is inequal to 1.")
        distribution_x.append(props)
    else:
        for state in gl_states[S[sValsIndex]]:
            sVals.append(state)
            dfs(x, S, sVals, sValsIndex+1, s_group, s_x_group, distribution_x)
            sVals.pop()
def getEntropy(x, pax, df):
    distribution_x = []
    paxName = []
    for pa in pax:
        paxName.append(df.columns[pa])
    xName = df.columns[x]
    paxAndXName = []
    paxAndXName = paxName.copy()
    paxAndXName.append(xName)
    if tuple(paxAndXName) in gl_group_cache:
        pax_x_group = gl_group_cache.get(tuple(paxAndXName))
    else:
        pax_x_group = df.groupby(paxAndXName)
        gl_group_cache[tuple(paxAndXName)] = pax_x_group
    if len(paxName) == 0:
        # no parents
        n = df.shape[0]
        props = []
        for state in gl_states[xName]:
            count = len(pax_x_group.get_group(state))
            props.append(count/n)
        distribution_x.append(props)
    else:
        if tuple(paxName) in gl_group_cache:
            pax_group = gl_group_cache.get(tuple(paxName))
        else:
            pax_group = df.groupby(paxName)
            gl_group_cache[tuple(paxName)] = pax_group
        dfs(x=xName, S=paxName, sVals=[], sValsIndex=0, s_group=pax_group, s_x_group=pax_x_group, distribution_x=distribution_x)
    return utils.H(utils.jema(distribution_x))
def dag_entropy(graph, df):
    if len(graph) != df.shape[1]: 
        print("dag_entropy: graph nodes is inconsistent with data")
        return -1
    nodes = df.shape[1]
    entropy = 0
    for x in range(nodes):
        #cache
        if gl_changed[gl_nodes[x]] == False:
            entropy += gl_entropy_history[gl_nodes[x]]
            continue
        pax = []
        for i in range(0, nodes):
            if graph[i][x] == 1:
                if (graph[x][i] == 1):
                    print("dag_entropy: the graph have undirected edge")
                    return -1
                pax.append(i)
        entropyX = getEntropy(x, pax, df)
        entropy += entropyX
        #cache
        gl_changed[gl_nodes[x]] = False
        gl_entropy_history[gl_nodes[x]] = entropyX
    return entropy

def getUndirected_edges(undirectedGraph):
    edges = 0
    for row in undirectedGraph:
        for val in row:
            if val == 1:
                edges+=1
    return edges/2
def search_all_dag(skeleton, undirect_edges:int, i, j, df, dag):    
    if undirect_edges == 0:
        dag2 = {"entropy": dag_entropy(skeleton, df), "graph": skeleton.copy()}
        if dag["entropy"] > dag2["entropy"]:
            dag["entropy"] = dag2["entropy"]
            dag["graph"] = dag2["graph"]
        dag["debug_count"] += 1
    else:
        row = len(skeleton)
        col = len(skeleton[0])
        while i<row:
            while j<col:
                if skeleton[i][j] == 1 and skeleton[j][i] == 1:
                    x = gl_nodes[i]
                    y = gl_nodes[j]
                    skeleton[i][j] = 1
                    skeleton[j][i] = 0
                    ii = i if j+1 < col else i+1
                    jj = j+1 if ii == i else 0
                    #cache
                    gl_changed[x] = gl_changed[y] = True
                    search_all_dag(skeleton, undirect_edges-1, ii, jj, df, dag)
                    skeleton[i][j] = 0
                    skeleton[j][i] = 1
                    #cache
                    gl_changed[x] = gl_changed[y] = True
                    search_all_dag(skeleton, undirect_edges-1, ii, jj, df, dag)
                    skeleton[i][j] = 1
                    skeleton[j][i] = 1
                    return
                j += 1
            i += 1
            j = 0
def init(df):
    global gl_nodes
    gl_nodes = list(df.columns)
    for node in gl_nodes:
        gl_changed[node] = True
        gl_entropy_history[node] = 0.0
    global gl_states
    gl_states = {}
    for colName in gl_nodes:
        gl_states[colName] = set()
        for state in df.get(colName):
            gl_states[colName].add(state)
    global gl_group_cache
    gl_group_cache = {}
def entropic_enumeration(df):
    init(df)
    # skeleton, O = utils.getSkeleton({"C": df.corr().values, "n": df.shape[0]})
    skeleton, O = utils.getSkeleton(df)
    undirected_edges = int(getUndirected_edges(skeleton))
    print("undirected_edges = %d" %undirected_edges)
    dag = {"graph":np.array([]), "entropy": 1e9+7, "debug_count":0}
    search_all_dag(skeleton, undirected_edges, 0, 0, df, dag)
    enumeration_edges = utils.narrayToEdges(dag['graph'], df)
    return utils.getAdjmat(enumeration_edges, df)
def debug():
    model = bn.import_DAG("asia")
    iterations = 10
    shd_sum = 0
    begintime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("begintime:%s" %(begintime))
    for i in range(0, iterations):
        #21条边10000 = 7h
        df = bn.sampling(model, n=10000)
        # print("columns : %s" %df.columns)
        correct_adjmat = model['adjmat']
        enumeration_adjmat = entropic_enumeration(df)
        shd = utils.SHD(correct_adjmat, enumeration_adjmat)
        # print(enumeration_adjmat)
        print("第%d次：" %(i+1))
        print("enumeration shd : %d" %shd)
        shd_sum += shd
    endtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("endtime:%s" %(endtime))
    print("avg shd %f" %(shd_sum/iterations))
# debug()