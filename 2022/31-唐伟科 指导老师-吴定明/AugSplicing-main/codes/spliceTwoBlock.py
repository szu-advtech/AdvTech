#此处的key是block2中指定维度指定属性值的项们
#该属性用于检查该项是否可以并入block1,可以并入的项需要满足在时间维度上在remainColsetdict，其他维度在existAttrSetdict中
def judge(key, existAttrSetdict, remainColsetdict, N, d):
    for idx in d:
        if key[idx] not in remainColsetdict[idx]:
            return False
    spModes = set(range(N)) - set(d)
    for idx in spModes:
        if key[idx] not in existAttrSetdict[idx]:
            return False
    return True

#判断这一项的这两个维度是否在block1和block2块中
def judge2(key, dimen1Attrs, dimen2Attrs, dimen1Attrs2, dimen2Attrs2, dimen1, dimen2):
    if key[dimen1] in dimen1Attrs and key[dimen2] in dimen2Attrs:
        return True
    if key[dimen1] in dimen1Attrs2 and key[dimen2] in dimen2Attrs2:
        return True
    return False


#找到两个块中相邻的那些项
def findInsecTuples(block1, block2,  existAttrSetdict, N):
    #Tuples以字典的形式存在，key为属性值，value为该项结果值
    block1Tuples, block2Tuples = block1.getTuples(), block2.getTuples()
    #block2ColKeysetDicts可以通过维度和该维度的属性值找到该维度属性中是输入值的项
    block2ColKeysetDicts = block2.getColKeysetDicts()
    #colKeysetDick取出来的是时间维度
    colKeysetDict = block2ColKeysetDicts[2]
    #cols是时间集
    cols = existAttrSetdict[N-1]
    isChanged, changeTuples = False, {}
    #找到紧贴这上一个时间片的项
    for col in cols:
        #找到时间属性为col的key集合
        keyset = colKeysetDict[col]
        for key in keyset:
            #下列判断语句作用：找到其他维度block1中的子块拥有，但时间维度值不同的项。
            if key not in block1Tuples and key[0] in existAttrSetdict[0] and key[1] in existAttrSetdict[1]:
               changeTuples[key] = block2Tuples[key]
    if len(changeTuples) != 0:
        isChanged = True
    return isChanged, changeTuples

#拼接某一模块
def spliceOnModes(block1, block2, existAttrSetdict, N, d):
    M, S, initden = block1.getMass(), block1.getSize(), block1.getDensity()
    m, isChanged  = len(d), False
    # modeToAttVals2是通过维度索引到当前的所有该维度的属性值
    modeToAttVals2 = block2.getAttributeDict()
    block2Tuples = block2.getTuples()
    block2ColKeysetDicts = block2.getColKeysetDicts()
    attrMassdict, newattrsKeydict, remainColsetdict = {}, {}, {}
    filColsetdict = filterBlock2Cols(initden, block2, existAttrSetdict, d)
    for idx in d:
        remainColset = modeToAttVals2[idx] - existAttrSetdict[idx]
        remainColsetdict[idx] = remainColset - filColsetdict[idx]
    mode = d[0]
    #remainColset:是需要拼接到block1中的块
    remainColset = remainColsetdict[mode]
    #colKeysetDict:是该维度中，属性值映射到对应的所有key的字典
    colKeysetDict = block2ColKeysetDicts[mode]
    for col in remainColset:
        #获取block2中符合目标维度列值为col的所有key
        keyset = colKeysetDict[col]
        for key in keyset:
            #通过判断则可以并入block1，添加新项需要新建attrMassdict
            if judge(key, existAttrSetdict, remainColsetdict, N, d):
                if m == 1:
                    attrkey = key[mode]
                else:
                    attrkey = tuple([key[dimen] for dimen in d])
                if attrkey not in attrMassdict:
                    attrMassdict[attrkey] = 0
                    newattrsKeydict[attrkey] = set()
                attrMassdict[attrkey] += block2Tuples[key]
                newattrsKeydict[attrkey].add(key)
    #按照新增加时间维度or其他维度的属性值的value进行排序
    sorted_dict = sorted(attrMassdict.items(), key=lambda x: x[1], reverse=True)
    accessColsetdict = {}
    for idx in range(m):
        accessColsetdict[idx] = set()
    for attr, mass in sorted_dict:
        #引用论文中的证明：M(x) > Q.g(Block1)
        if mass >= m * initden:
            #newattrsKeydict:可以通过指定的属性值找到需要添加的keys
            attrkeys = newattrsKeydict[attr]
            changeTuples = {}
            for product in attrkeys:
                changeTuples[product] = block2Tuples[product]
            block1.addUpdate(changeTuples)
            block2.removeUpdate(changeTuples)
            initden = block1.getDensity()
            isChanged = True
        else:
            break
    return isChanged, block1, block2


#更新block1、block2操作
# should update block1, block2
#对应论文的拼接b
def alterCalOneModeByMost(block1, block2, existAttrSetdict, newAttrSetdict, attrMassdict,
                          attrTupledict, N, idx, isfirst):
    M, S, initden = block1.getMass(), block1.getSize(), block1.getDensity()
    modeToAttVals2 = block2.getAttributeDict()
    block2Tuples = block2.getTuples()
    block2ColKeysetDicts = block2.getColKeysetDicts()
    colKeysetDict = block2ColKeysetDicts[idx]
    newattrs, isChanged = set(), False

    #remainCols:存储block2还剩下的属性
    remainCols = modeToAttVals2[idx] - existAttrSetdict[idx]
    filterColset = filterBlock2Cols(initden, block2, existAttrSetdict, d=[idx])[idx]
    #去除filterColset中的属性值
    remainCols.difference_update(filterColset)
    cols = list(range(N))
    #过滤掉没有共同属性值的属性
    cols.remove(idx)
    dimen1, dimen2 = cols[0], cols[1]
    if isfirst:
        dimen1Attrs = newAttrSetdict[dimen1] | existAttrSetdict[dimen1]
        dimen2Attrs = newAttrSetdict[dimen2] | existAttrSetdict[dimen2]
        dimen1Attrs2, dimen2Attrs2 = [], []
    else:
        if len(newAttrSetdict[dimen1]) == 0 and len(newAttrSetdict[dimen2]) != 0:
            dimen1Attrs = existAttrSetdict[dimen1]
            dimen2Attrs = newAttrSetdict[dimen2]
            dimen1Attrs2, dimen2Attrs2 = [], []
        elif len(newAttrSetdict[dimen1]) != 0 and len(newAttrSetdict[dimen2]) == 0:
            dimen1Attrs = newAttrSetdict[dimen1]
            dimen2Attrs = existAttrSetdict[dimen2]
            dimen1Attrs2, dimen2Attrs2 = [], []
        elif len(newAttrSetdict[dimen1]) != 0 and len(newAttrSetdict[dimen2]) != 0:
            dimen1Attrs = existAttrSetdict[dimen1] | newAttrSetdict[dimen1]
            dimen2Attrs = newAttrSetdict[dimen2]
            dimen1Attrs2 = newAttrSetdict[dimen1]
            dimen2Attrs2 = existAttrSetdict[dimen2]
        else:
            dimen1Attrs, dimen2Attrs, dimen1Attrs2, dimen2Attrs2 = [], [], [], []
    remainColsMassdict = {}

    for col in remainCols:
        keyset = colKeysetDict[col]
        for key in keyset:
            if key not in attrTupledict[col]:
                belong = judge2(key, dimen1Attrs, dimen2Attrs, dimen1Attrs2, dimen2Attrs2, dimen1, dimen2)
                if belong:
                    attrMassdict[col] += block2Tuples[key]
                    attrTupledict[col].add(key)
        remainColsMassdict[col] = attrMassdict[col]

    #按照每一项的mass排序
    sorted_dict = sorted(remainColsMassdict.items(), key=lambda x: x[1], reverse=True)
    for attr, mass in sorted_dict:
        if mass >= initden:
            M = M + mass
            S = S + 1
            initden = M / S
            newattrs.add(attr)
        else:
            break
    changeTuples = {}
    if len(newattrs) != 0:
        for attr in newattrs:
            attrkeys = attrTupledict[attr]
            for product in attrkeys:
                changeTuples[product] = block2Tuples[product]
    if len(changeTuples) != 0:
        isChanged = True
        block1.addUpdate(changeTuples)
        block2.removeUpdate(changeTuples)
    return isChanged, block1, block2, attrMassdict, newattrs, attrTupledict

#existAttrSetdict存储了两个块中每个维度的共同属性, initden是block1的密度，d是没有属性重叠的维度
#注意：该函数添加了时间维度的子块后，并没有将这些块移出block2
#该函数的作用：过滤掉没必要拼接属性值，即添加到b1块中会使b1块密度变小的属性
def filterBlock2Cols(initden, block2, existAttrSetdict, d):
    modeToAttVals2 = block2.getAttributeDict()
    block2ColDegreeDicts = block2.getColDegreeDicts()
    filterColsetdict = {}
    M2, initden2 = block2.getMass(), block2.getDensity()
    for idx in d:
        #找出block2块中有而block1块中没有的属性值
        remainCols = modeToAttVals2[idx] - existAttrSetdict[idx]
        filterColsetdict[idx] = set()
        #某一维度的size
        collen = len(modeToAttVals2[idx])
        #thres是idx维度中的一片
        thres = M2 - (collen - 1) * initden2
        if thres < initden:
            #filterColsetdict存储的是该维度需要过滤掉的块
            filterColsetdict[idx] = remainCols
        else:
            block2ColDegreeDict = block2ColDegreeDicts[idx]
            for col in remainCols:
                degree = block2ColDegreeDict[col]
                if degree < initden:
                    filterColsetdict[idx].add(col)
    return filterColsetdict

#existsAttrSetdict是block1、block2中共同拥有的属性
def alterCalModesByMost(block1, block2, existAttrSetdict, N):
    modeToAttVals2 = block2.getAttributeDict()
    'initializa attrMassdicts, newAttr, existAttr'
    newAttrSetdict, attrMassdicts, attrTupledicts = {}, {}, {}
    for idx in range(N):
        newAttrSetdict[idx] = set()
        attrMassdicts[idx] = {}
        attrTupledicts[idx] = {}
        for attr in modeToAttVals2[idx]:
            attrMassdicts[idx][attr] = 0
            attrTupledicts[idx][attr] = set()
    isContinue, isfirst = True, True
    while isContinue:
        isContinue = False
        for idx in range(N):
            attrMassdict = attrMassdicts[idx]
            attrTupledict = attrTupledicts[idx]
            'when recur to this dimension again, existing cols should change'
            existAttrSetdict[idx] = existAttrSetdict[idx] | newAttrSetdict[idx]
            isChanged, block1, block2, attrMassdicts[idx], newAttrSetdict[idx], attrTupledicts[idx] = \
                alterCalOneModeByMost(block1, block2, existAttrSetdict, newAttrSetdict,
                                      attrMassdict, attrTupledict, N, idx, isfirst)
            if isChanged:
                if block2.getSize() == 0:
                    isContinue = False
                    break
                isContinue = True
        if isContinue:
            isfirst = False
    return block1, block2


#拼接两个子块
def splice_two_block(block1, block2, N):
    sflag, isChanged = False, False
    modeToAttVals1 = block1.getAttributeDict()
    modeToAttVals2 = block2.getAttributeDict()
    'existAttrSetdict: attributes of block2 which have taken'
    d, insec_dimens_dict = [], {}
    for idx in range(N):
        #找出两个块中该维度重叠的值
        insec_dimens = modeToAttVals1[idx] & modeToAttVals2[idx]
        if len(insec_dimens) == 0:
            #d数组用来存储没有重叠属性值的维度
            d.append(idx)
        insec_dimens_dict[idx] = insec_dimens
    if len(d) == N:
        #sflag表示是否拼接
        return sflag, block1, block2
    else:
        'dimension->dict(col->mass)'
        #两个子块的所有维度都有交叉的地方才可拼接
        #这是第一种情况，当b2块中所有属性b1块都有，但是具体的项b1却没有则直接将这些块加入b1,从b2去除。
        if len(d) == 0:
            #找到block2块中可拼接如block1的项，isChanged是判断是否找到目标子块
            'add insec block into block1, remove it from block2'
            #findInsecTuples():函数作用是将block2中和block1重叠的项拼接给block1
            isChanged, changeTuples = findInsecTuples(block1, block2, insec_dimens_dict, N)
            block1.addUpdate(changeTuples)
            block2.removeUpdate(changeTuples)
        if len(d) >= 1:
            'only calculate laped block'
            # 只在一个维度上拼接,为论文的a步骤，拼接时间模块
            isChanged, block1, block2 = \
                spliceOnModes(block1, block2, insec_dimens_dict, N, d)
        if isChanged:
            sflag = True
            # block1、bloack2发生了改变，但是block2仍然不为空
            if block2.getSize() != 0:
                for idx in range(N):
                    insec_dimens_dict[idx] = modeToAttVals1[idx] & modeToAttVals2[idx]
                block1, block2 = alterCalModesByMost(block1, block2, insec_dimens_dict, N)
        return sflag, block1, block2

