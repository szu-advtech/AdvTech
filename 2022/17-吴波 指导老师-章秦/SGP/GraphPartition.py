import networkx as nx
import math


# 边权计算
def computeEW(G):
    for e in G.edges:
        # 边权值
        EW = 0

        N_e = 0

        # 分组e的两个端点的邻居
        V_u = []
        V_v = []
        V_uv = []

        for item in G.adj[e[0]]:
            if item != e[1]:
                V_u.append(item)

        for item in G.adj[e[1]]:
            if item != e[0]:
                V_v.append(item)

        # 查找共同邻居
        for i in V_u:
            for k in V_v:
                if i == k:
                    V_uv.append(i)

        for i in V_uv:
            V_u.remove(i)
            V_v.remove(i)

        if len(V_u) + len(V_v) + len(V_uv) != 0:
            EW = + len(V_uv) / (len(V_u) + len(V_v) + len(V_uv))

        # 计算R(V_u,V_uv)
        N_e = 0
        for i in V_u:
            for k in V_uv:
                if (G.has_edge(i, k)):
                    N_e += 1
        if len(V_u) * len(V_uv) != 0:
            EW += N_e / (len(V_u) * len(V_uv))
            G[e[0]][e[1]]['R(V_u,V_uv)'] = N_e / (len(V_u) * len(V_uv))
        else:
            G[e[0]][e[1]]['R(V_u,V_uv)'] = 0

        # 计算R(V_u,V_v)
        N_e = 0
        for i in V_u:
            for k in V_v:
                if (G.has_edge(i, k)):
                    N_e += 1
        if len(V_u) * len(V_v) != 0:
            EW += N_e / (len(V_u) * len(V_v))
            G[e[0]][e[1]]['R(V_u,V_v)'] = N_e / (len(V_u) * len(V_v))
        else:
            G[e[0]][e[1]]['R(V_u,V_v)'] = 0

        # 计算R(V_uv,V_v)
        N_e = 0
        for i in V_uv:
            for k in V_v:
                if (G.has_edge(i, k)):
                    N_e += 1
        if len(V_uv) * len(V_v) != 0:
            EW += N_e / (len(V_uv) * len(V_v))
            G[e[0]][e[1]]['R(V_uv,V_v)'] = N_e / (len(V_uv) * len(V_v))
        else:
            G[e[0]][e[1]]['R(V_uv,V_v)'] = 0

        # 计算R(V_uv)
        N_e = 0
        for i in V_uv:
            for k in V_uv:
                if G.has_edge(i, k):
                    N_e += 1
        N_e /= 2
        if len(V_uv) * (len(V_uv) - 1) != 0:
            EW += 2 * N_e / (len(V_uv) * (len(V_uv) - 1))
            G[e[0]][e[1]]['R(V_uv)'] = 2 * N_e / (len(V_uv) * (len(V_uv) - 1))
        else:
            G[e[0]][e[1]]['R(V_uv)'] = 0

        G[e[0]][e[1]]['EW'] = EW
        G[e[0]][e[1]]['V_uv'] = V_uv
        G[e[0]][e[1]]['V_u'] = V_u
        G[e[0]][e[1]]['V_v'] = V_v


# 过滤边
def graphPartition(G):
    computeEW(G)

    # 阈值计算
    EWset = set([])
    minEW = math.inf
    maxEW = -1
    for e in G.edges:
        EWset.add(G[e[0]][e[1]]['EW'])
    for i in EWset:
        if i < minEW:
            minEW = i
        if i > maxEW:
            maxEW = i

    T = minEW + (maxEW - minEW) * 0.2

    # print(EWset)
    # print(f"{minEW}---{maxEW}")

    # 过滤边
    GPartition = G.copy()
    for e in GPartition.edges:
        if GPartition[e[0]][e[1]]['EW'] < T:
            GPartition.remove_edge(e[0], e[1])

    return GPartition


def DFS(G, subGraph, v):
    G.nodes[v]['visited'] = 1
    subGraph.add_node(v)

    for u in G.adj[v]:
        if G.nodes[u]['visited'] == 0:
            DFS(G, subGraph, u)
