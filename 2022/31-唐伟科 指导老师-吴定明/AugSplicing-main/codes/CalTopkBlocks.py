import random
import networkx as nx
import codes.spliceTwoBlock as stb


# #按照密度进行块的插入
def insertBlockbyDensity(block, blocklist):
    density = block.getMass() / block.getSize()
    canfind = False
    #从头到尾遍历已有的块，这些块是按密度从大到小排序的。找到第一个比当前块小的块，插入该块到这个小的块的位置。
    for idx, block2 in enumerate(blocklist):
        density2 = block2.getMass() / block2.getSize()
        if density >= density2:
            blocklist.insert(idx, block)
            canfind = True
            break
    #处理边界条件，若插入的快比列表中的块密度都要小，则插入至尾巴。
    if not canfind:
        blocklist.append(block)
    return blocklist

# 二分查找插入策略
# def insertBlockbyDensity(block, blocklist):
#     density = block.getMass() / block.getSize()
#     left, right = 0, len(blocklist) - 1
#     while left <= right:
#         mid = int(left + (right - left) / 2)
#         midBlockDen = blocklist[mid].getMass() / blocklist[mid].getSize()
#         if midBlockDen > density:
#             left = mid + 1
#         else:
#             right = mid - 1
#     if left == len(blocklist):
#         blocklist.append(block)
#     else:
#         blocklist.insert(left, block)
#     return blocklist

#得到最终的k + l个密集子张量
def getResultBlocks(blockNumdict, k, l):
    results = []
    for key in blockNumdict:
        block = blockNumdict[key]
        if block.getSize() == 0:
            continue
        insertBlockbyDensity(block, results)
    return results[:k+l]

#初始化函数：创建一个图，图的边表示了两个新旧密集块有共同的属性值
def init_graph(blocklist1, blocklist2, N, k):
    edges = []
    blockNumdict = {}
    #这次为边密度字典
    edgesDensity = {}
    #将块映射到字典中
    for idx, block in enumerate(blocklist1):
        blockNumdict[idx] = block
    for idx, block in enumerate(blocklist2):
        blockNumdict[idx + k] = block
    for idx1, block1 in enumerate(blocklist1):
        #modeToAttVals1通过维度可映射出属性值
        modeToAttVals1 = block1.getAttributeDict()
        for idx2, block2 in enumerate(blocklist2):
            modeToAttVals2 = block2.getAttributeDict()
            for dimen in range(N):
                #求出两个块中某个维度共有的属性值
                insec_dimens = modeToAttVals1[dimen] & modeToAttVals2[dimen]
                if len(insec_dimens) != 0:
                    #edges存入的数据表示两个块有共同的属性
                    edges.append((idx1, k+idx2))
                    #将每条边的密度和对应的两个顶点使用字典建立映射关系
                    density = block1.getDensity() + (block2.getDensity() - block1.getDensity()) / 2
                    edgesDensity[(idx1, k+idx2)] = density
                    break
    #建立一个图
    G = nx.Graph()
    #将edgesDensity按照值从大到小进行排序
    edgesDensity = sorted(edgesDensity.items(), key=lambda x:x[1], reverse=True)
    #读取edges中的所有边，输入图中
    G.add_edges_from(edges)
    return G, blockNumdict, edgesDensity

# def init_graph(blocklist1, blocklist2, N, k):
#     edges, blockNumdict = [], {}
#     for idx1, block1 in enumerate(blocklist1):
#         blockNumdict[idx1] = block1
#     for idx2, block2 in enumerate(blocklist2):
#         blockNumdict[idx2 + k] = block2
#     for idx1, block1 in enumerate(blocklist1):
#         b1mode2Attr = block1.getAttributeDict()
#         for idx2, block2 in enumerate(blocklist2):
#             b2mode2Attr = block2.getAttributeDict()
#             for i in range(N):
#                 shareAttr = b1mode2Attr[i] & b2mode2Attr[i]
#                 if len(shareAttr) != 0:
#                     edges.append((idx1, k + idx2))
#                     break
#     G = nx.Graph()
#     G.add_edges_from(edges)
#     return G, blockNumdict


'block1: remove edge with taken_ego_nodes, add edge with left_ego_nodes'
'block2: remove edge with left_ego_nodes'
def update_graph(G, blockNumdict,  takennode, leftnode, N):
    takenblock = blockNumdict[takennode]
    leftblock = blockNumdict[leftnode]
    modeToAttVals1 = takenblock.getAttributeDict()
    modeToAttVals2 = leftblock.getAttributeDict()
    left_ego_nodes = set(list(nx.ego_graph(G, leftnode, radius=1)))
    taken_ego_nodes = set(list(nx.ego_graph(G, takennode, radius=1)))
    'block1: add edge with left_ego_nodes'
    #添加被拼接块与其他块的边，nodeset1是与被拼接块相连的其他块序号。一旦这些块与拼接后的块在任何维度
    #有相同的属性值，则建立与新块的新边
    nodeset1 = left_ego_nodes - taken_ego_nodes
    for node in nodeset1:
        block = blockNumdict[node]
        modeToAttVals3 = block.getAttributeDict()
        for dimen in range(N):
            insec_dimens = modeToAttVals1[dimen] & modeToAttVals3[dimen]
            if len(insec_dimens) != 0:
                G.add_edge(node, takennode)
                break
    #taken_ego_nodes是与拼接块（block1）相邻的节点。除去原点和被拼接点
    'block1: remove edge with taken_ego_nodes'
    taken_ego_nodes.remove(takennode)
    taken_ego_nodes.remove(leftnode)
    for node in taken_ego_nodes:
        block = blockNumdict[node]
        modeToAttVals3 = block.getAttributeDict()
        #insec用于判断原本的图中的邻接块是否与block1有相同的属性值。若没有，则使用G.remove_edge消除两者的边
        insec = False
        for dimen in range(N):
            insec_dimens = modeToAttVals1[dimen] & modeToAttVals3[dimen]
            if len(insec_dimens) != 0:
                insec = True
                break
        if not insec:
            G.remove_edge(node, takennode)
    'block2: remove edge with left_ego_nodes'
    #left_ego_nodes是被拼接块block2的邻接点，一下程序的作用是：检查原本的邻接块是否仍然与剩下的块有共同
    # 属性，若没有共同属性则将两者在图中的边删除
    left_ego_nodes.remove(leftnode)
    for node in left_ego_nodes:
        block = blockNumdict[node]
        modeToAttVals3 = block.getAttributeDict()
        insec = False
        for dimen in range(N):
            insec_dimens = modeToAttVals2[dimen] & modeToAttVals3[dimen]
            if len(insec_dimens) != 0:
                insec = True
                break
        if not insec:
            G.remove_edge(node, leftnode)
    return G


#该函数的作用：1、block1拼接块添加新的边，删除没有关联的边。2、删除图中block2节点，自然与该顶点的边全部删除
'block1: remove edge with taken_ego_nodes, add edge with left_ego_nodes'
'block2: delete leftnode'
def remove_update_graph(G, blockNumdict,takennode, leftnode, N):
    takenblock = blockNumdict[takennode]
    modeToAttVals1 = takenblock.getAttributeDict()
    left_ego_nodes = set(list(nx.ego_graph(G, leftnode, radius=1)))
    taken_ego_nodes = set(list(nx.ego_graph(G, takennode, radius=1)))
    nodeset1 = left_ego_nodes - taken_ego_nodes
    'block1: add edge with left_ego_nodes'
    for node in nodeset1:
        block = blockNumdict[node]
        modeToAttVals3 = block.getAttributeDict()
        for dimen in range(N):
            insec_dimens = modeToAttVals1[dimen] & modeToAttVals3[dimen]
            if len(insec_dimens) != 0:
                G.add_edge(node, takennode)
                break
    'block1: remove edge with taken_ego_nodes'
    taken_ego_nodes.remove(takennode)
    taken_ego_nodes.remove(leftnode)
    for node in taken_ego_nodes:
        block = blockNumdict[node]
        modeToAttVals3 = block.getAttributeDict()
        insec = False
        for dimen in range(N):
            insec_dimens = modeToAttVals1[dimen] & modeToAttVals3[dimen]
            if len(insec_dimens) != 0:
                insec = True
                break
        if not insec:
            G.remove_edge(node, takennode)
    'delete leftnode'
    G.remove_node(leftnode)
    return G


#核心函数，计算出topk个密集子块
def calTopkBlocks(blocklist1, blocklist2, k, l, maxSp, N):
    #此时输出的edgesDensity为列表
    G, blockNumdict, edgesDensity = init_graph(blocklist1, blocklist2, N, k)
    #[((0, 10), 3.320209194214876), ((1, 10), 2.872217188463911), ((0, 11), 2.803431591928474), ((5, 10), 2.5824752697841724), ((6, 10), 2.5824752697841724), ((7, 10), 2.5824752697841724), ((8, 10), 2.5824752697841724), ((9, 10), 2.5824752697841724)]
    if len(G.nodes) == 0:
        return getResultBlocks(G, blockNumdict, k, l)
    #lastden：存储新旧张量中最小密度的子块
    lastden = min(blocklist1[-1].getDensity(), blocklist2[-1].getDensity())
    i, fail, index = 0, 0, 0
    #最多的拼接次数为maxSp，未拼接次数超过5 * maxSp也推出程序，当索引index超出edgesDensity时也退出拼接
    while i < maxSp and fail < 5 * maxSp and index < len(edgesDensity):
        #随机选择G中的某一条边，取出该边关联的两个块block1、block2
        cnode, neighnode = random.choice(list(G.edges()))
        cnode, neighnode = edgesDensity[index][0]
        density = edgesDensity[index][1]
        index += 1
        if density < lastden or G.__contains__(cnode) == False or G.__contains__(neighnode) == False:
            continue
        block1 = blockNumdict[cnode]
        block2 = blockNumdict[neighnode]
        #确保使用密度小的块拼接密度大的块
        if block1.getDensity() >= block2.getDensity():
            sflag, takenBlock, leftBlock = stb.splice_two_block(block1, block2, N)
            label = '1'
        else:
            sflag, takenBlock, leftBlock = stb.splice_two_block(block2, block1, N)
            label = '2'
        #sflag判断是否发生过拼接
        if sflag:
            i += 1
            if label == '2':
                cnode, neighnode = neighnode, cnode
            #更新块列表中的块为拼接后的块
            blockNumdict[cnode], blockNumdict[neighnode] = takenBlock, leftBlock
            #如果拼接剩下的块size不为0且密度比最小密度还小执行update_graph函数。留下的块要是比原本的最小
            # 密度的块密度还小或者全部被拼接到拼接块中了，则可以删除其在图中的顶点和边。
            if leftBlock.getSize() != 0 and leftBlock.getDensity() >= lastden:
                'update rule changes'
                G = update_graph(G, blockNumdict, cnode, neighnode, N)
            else:
                G = remove_update_graph(G, blockNumdict, cnode, neighnode, N)
        else:
            fail += 1
    return getResultBlocks(blockNumdict, k, l)

# def calTopkBlocks(blocklist1, blocklist2, k, l, maxSp, N):
#     G, blockNumdict = init_graph(blocklist1, blocklist2, N, k)
#     if len(G.nodes) == 0:
#         return getResultBlocks(blockNumdict, k, l)
#     lastden = min(blocklist1[-1].getDensity(), blocklist2[-1].getDensity())
#     i, fail = 0, 0
#     while i < maxSp and fail < 5 * maxSp:
#         node, neighbor = random.choice(list(G.edges()))
#         block1 = blockNumdict[node]
#         block2 = blockNumdict[neighbor]
#         if block1.getDensity() > block2.getDensity():
#             sflag, takenBlock, leftBlock = stb.splice_two_block(block1, block2, N)
#             lable = '1'
#         else:
#             sflag, takenBlock, leftBlock = stb.splice_two_block(block2, block1, N)
#             lable = '2'
#         if sflag:
#             i += 1
#             if lable == '2':
#                 node, neighbor = neighbor, node
#             blockNumdict[node] = takenBlock
#             blockNumdict[neighbor] = leftBlock
#             if leftBlock.getSize() != 0 and leftBlock.getDensity() > lastden:
#                 G = update_graph(G, blockNumdict, node, neighbor, N)
#             else:
#                 G = remove_update_graph(G, blockNumdict, node, neighbor, N)
#         else:
#             fail += 1
#     return getResultBlocks(blockNumdict, k, l)