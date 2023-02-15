import os
import codes.blockClass as bloc


def delAttri(path):
    for name in os.listdir(path):
        if name.endswith(".attributes"):
            os.remove(os.path.join(path, name))


def calFscore(predset, trueset):
    correct = 0
    for p in predset:
        if p in trueset: correct += 1
    pre = 0 if len(predset) == 0 else float(correct) / len(predset)
    rec = 0 if len(trueset) == 0 else float(correct) / len(trueset)
    print('pre: {}, rec: {}'.format(pre, rec))
    if pre + rec > 0:
        F = 2 * pre * rec / (pre + rec)
    else:
        F = 0
    print('f1:{}'.format(F))
    return F


def cal_block_density(file):
    ids = set()
    apps = set()
    it_sts = set()
    mass = 0
    f = open(file, 'r')
    for line in f.readlines():
        cols = line.replace('\n', '').split(',')
        ids.add(cols[0])
        apps.add(cols[1])
        it_sts.add(cols[2])
        mass = mass + float(cols[3])
    blocksize = ids.__len__()+apps.__len__()+it_sts.__len__()
    if blocksize != 0:
        density = round(mass/blocksize, 1)
    else:
        density = 0
    return density, mass, blocksize

#将block写入tuplegilename文件中
def writeBlockToFile(path, block, tuplefilename):
    tuplefile = os.path.join(path, tuplefilename)
    tuples = block.getTuples()
    print("Size: " + str(block.getSize()))
    print("Mass: " + str(block.getMass()))
    print("Density: " + str(block.getDensity()))
    with open(tuplefile, 'w') as tuplef:
        for key in tuples:
            words = list(key)
            words.append(str(tuples[key]))
            tuplef.write(','.join(words))
            tuplef.write('\n')
    tuplef.close()

#从path中读取指定k个块,块的类型由bloackClass定义，返回一个块列表。
def readBlocksfromPath(path, k):
    blocklist = []
    for i in range(1, k+1):
        tuplefile = os.path.join(path, 'block_{}.tuples'.format(i))
        block = readBlock(tuplefile)
        blocklist.append(block)
    return blocklist

#从指定文件中读取信息并建立块返回
def readBlock(tuplefile):
    attridict, colDegreeDicts, colKeysetDicts, tupledict = {}, {}, {}, {}
    dimension = 3
    for idx in range(dimension):
        attridict[idx] = set()
        colDegreeDicts[idx] = {}
        colKeysetDicts[idx] = {}
    M = 0.0
    with open(tuplefile, 'r') as tf:
        for line in tf:
            #取出每一行数据存入cols列表
            cols = line.strip().split(',')
            #分割数据，前三列为key，最后一列为value
            key, value = tuple(cols[:-1]), int(cols[-1])
            #数据整合
            if key not in tupledict:
                tupledict[key] = value
            else:
                tupledict[key] += value
            #M为该块的质量，为块中每一项的值的总和
            M += value
            for idx in range(dimension):
                attr = key[idx]
                if attr not in attridict[idx]:
                    #attridict是通过维度索引到当前的所有该维度的属性值
                    attridict[idx].add(attr)
                    #colkeysetDicts是一个二维矩阵，通过维度和该维度的属性值找到在idx维度中值是attr的所有项的key
                    colKeysetDicts[idx][attr] = set()
                    colKeysetDicts[idx][attr].add(key)
                    #colDegreeDicts是一个二维矩阵，计算idx维度下属性值为attr的值的总和
                    colDegreeDicts[idx][attr] = value
                else:
                    colDegreeDicts[idx][attr] += value
                    colKeysetDicts[idx][attr].add(key)
    size = len(attridict[0]) + len(attridict[1]) + len(attridict[2])
    block = bloc.block(tupledict, attridict, colDegreeDicts, colKeysetDicts, M, size, dimension)
    return block


def saveSimpleListData(simls, outfile):
    with open(outfile, 'w') as fw:
        'map(function, iterable)'
        fw.write('\n'.join(map(str, simls)))
        fw.write('\n')
        fw.close()


# if __name__ == '__main__':
#     block = readBlock('/home/tangweike/PycharmProjects/AugSplicing/codes/augfile_dcube_output/0/block_1.tuples')
#     print(block.getSize())
#     print(block.getMass())
#     print(block.getDensity())
#     print(block.getTuples())
#     print(block.getAttributeDict())
#     print(block.getColDegreeDicts())
#     print(block.getColKeysetDicts())




