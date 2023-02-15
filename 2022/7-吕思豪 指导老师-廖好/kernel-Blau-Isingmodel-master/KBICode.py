import sys

import time
import numpy as np
from math import *
import copy
import random
from math import sqrt, exp, log
import operator as op
import bisect


def cdf(weights):
    total = sum(weights)
    result = []
    cumsum = 0
    for w in weights:
        cumsum += w
        result.append(cumsum / total)
    return result


def choice(population, cumm):
    # assert len(population) == len(weights)
    # cdf_vals = cumm
    x = random.random()
    idx = bisect.bisect(cumm, x)
    return population[idx]


def likelihood(KKH, aarea, seta, ppm, userw, JJ):    # seta: 自旋方向
    like = 0.
    l1 = 0.
    l2 = 0.
    for i in seta.keys():    # Hamiltonian function
        cor = userw[i]
        for j in ppm[i]:
            l1 = l1 + seta[i] * seta[j] * 0.5
        for k in range(len(KKH) - 1):
            for l in range(len(KKH[k])):
                l2 = l2 + KKH[k][l] * aarea[cor][1][k][l] * seta[i]
        l2 = l2 + KKH[3][0] * aarea[cor][1][4][0] * seta[i]
    l1 = l1 * JJ
    like = l1 + l2
    return like, l1, l2


def updatelike(KKH, aarea, seta, ppm, L, x, ppair, userw, JJ):
    old1 = 0.
    old2 = 0.
    new = 0.
    os = seta[x]
    cor = userw[x]

    for e in ppm[x]:
        old1 = old1 + os * seta[e]
    for k in range(len(KKH) - 1):
        for l in range(len(KKH[k])):
            old2 = old2 + KKH[k][l] * aarea[cor][1][k][l] * os
    old2 = old2 + KKH[3][0] * aarea[cor][1][4][0] * os
    old1 = old1 * JJ
    L2 = L - 2. * (old1 + old2)
    return L2


def generateAgg(K, Kparams, aarea, ppair):
    Agg = {}
    llen = aarea.keys()    # area.keys:wardCode
    A = 0.  # estimated number of links
    tnodes = 0
    for ii in range(len(llen)):
        i = llen[ii]
        s1 = size[i]
        tnodes = tnodes + s1
        for jj in range(len(llen) - (ii)):
            kk = ii + jj
            k = llen[kk]
            s2 = size[k]
            dgg = Kparams[0]
            for zz in range(K):
                dgg = dgg + Kparams[zz + 1] * ppair[(i, k)][zz]
            if i != k:
                A = A + s1 * s2 * exp(-1. * (dgg))
            else:
                A = A + s1 * (s1 - 1.) * 0.5 * exp(-1. * (dgg))
            Agg[str(i) + str(k)] = 1. / (1 + exp(dgg))
            Agg[str(k) + str(i)] = 1. / (1 + exp(dgg))
    return Agg, A / float(tnodes)


def generatenet(Agg, aarea, userw, size):
    llen = aarea.keys()    # area.keys:wardCode
    llenu = {}
    llindex = []
    ssize = 0
    for w in llen:
        llenu[w] = []
        llindex.append(ssize)
        ssize = ssize + size[w]
    users = len(userw.keys())
    net = {}
    for i in range(users):
        net[i] = []
    totl = 0
    for ii in range(len(llen)):
        cor = llen[ii]
        indi = llindex[llen.index(cor)]
        pp = Agg[str(cor) + str(cor)]
        Ni = size[cor]
        # without repetion
        for i in range(Ni):
            for j in range(Ni - (i + 1)):
                k = i + j + 1
                if random.random() < pp:
                    net[indi + i].append(indi + k)
                    net[indi + k].append(indi + i)
                    totl = totl + 1

        for jj in range(len(llen) - (ii + 1)):
            kk = ii + jj + 1
            cor2 = llen[kk]
            indk = llindex[llen.index(cor2)]
            pp = Agg[str(cor) + str(cor2)]
            # with repetition
            Nk = size[cor2]
            for i in range(Ni):
                for k in range(Nk):
                    if random.random() < pp:
                        net[indi + i].append(indk + k)
                        net[indk + k].append(indi + i)
                        totl = totl + 1
    return net, totl / float(users)


def GgenerateSpins(KKH, userw, Kparams, ppm, aarea, ppair, JJ):    # ppm: net网络结构
    # random network:
    spins = [-1, 1]
    nseta = {}     # nseta: 自旋方向
    users = len(userw.keys())
    for e in range(users):
        nseta[e] = random.choice(spins)
    count = 0
    L, gg1, gg2 = likelihood(KKH, aarea, nseta, ppm, userw, JJ)   # 计算Hamiltonian function
    i = 0
    ext_time = 0
    cacc = 0
    T = 1.
    llen = aarea.keys()
    while count < 55 * users:    # 修正Hamiltonian function
        x = random.choice(range(users))
        # evalueta change of likelihood:
        L2 = updatelike(KKH, aarea, nseta, ppm, L, x, ppair, userw, JJ)

        AL = L - L2
        try:
            th = 1. / (1. + exp(AL / T))
        except OverflowError:
            th = exp(-AL / T)
            pass
        a = random.random()
        if a <= th:
            nseta[x] = nseta[x] * (-1)
            L = L2

            cacc = cacc + 1
        else:
            count = count + 1
        ext_time += 1
    L, ll1, ll2 = likelihood(KKH, aarea, nseta, ppm, userw, JJ)
    return nseta, ll1, ll2


def latlongdistance(lat1, lon1, lat2, lon2):
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance


def ncr(n, r):
    r = min(r, n - r)
    if r == 0: return 1
    numer = reduce(op.mul, xrange(n, n - r, -1))
    denom = reduce(op.mul, xrange(1, r + 1))
    return numer // denom


def funcmean(vec):
    mean = sum(vec) / float(len(vec))
    s2 = 0.
    for e in vec:
        s2 = s2 + (e - mean) * (e - mean)
    s2 = sqrt(s2 / (len(vec) - 1.))
    return mean, s2


def areadistance(K, KH, KH12, BKH, Kparams, aarea, Baarea, inBor, ppair, size, userw, J, J12, BJ, ggg):
    # genarate Agg: 生成邻接矩阵
    Agg, AA = generateAgg(K, Kparams, aarea, ppair)
    if AA < 3:  # We only generate spin configurations for networks with average degree (<K>) smalles than 6 (<K>=2L/N; AA=L/N)
        # same network for the three elections: 生成网络结构
        net, avel = generatenet(Agg, aarea, userw, size)

        # ME16:
        fout2 = open('ME16config' + str(ggg) + '.dat', 'w')
        fout2.write('wards KBIfraction originalfration\n')
        nseta, ml1, ml2 = GgenerateSpins(KH, userw, Kparams, net, aarea, ppair, J)  # 生成自旋
        llen = aarea.keys()    # area.keys:wardCode
        llindex = []
        ssize = 0
        for w in llen:
            llindex.append(ssize)
            ssize = ssize + size[w]
        dall = 0.
        tall = 0
        di = 0.
        tutto = 0
        for cor in aarea.keys(): # area.keys:wardCode
            if cor in Baarea.keys():
                indi = llindex[llen.index(cor)]
                rneg = aarea[cor][0][0]    # area: { wardCode:[ [ME16,ME12],[education,age,gender,[longitude,latitude],income],BoroughID ] }
                neg = 0
                for e in range(size[cor]):
                    if nseta[indi + e] == -1:
                        neg = neg + 1
                neg = neg / float(size[cor])
                dall = dall + abs(neg - rneg)
                tall = tall + 1
                di = di + abs(neg - rneg)
                tutto = tutto + 1
                fout2.write('%s %s %s\n' % (cor, neg, rneg))
            else:
                indi = llindex[llen.index(cor)]
                rneg = aarea[cor][0][0]
                neg = 0
                for e in range(size[cor]):
                    if nseta[indi + e] == -1:
                        neg = neg + 1
                neg = neg / float(size[cor])
                dall = dall + abs(neg - rneg)
                tall = tall + 1
                fout2.write('%s %s %s\n' % (cor, neg, rneg))
        d2 = 0
        t2 = 0
        for a in inBor.keys():
            neg = 0
            asize = 0
            rneg = inBor[a][0][0]
            for cor in inBor[a][1]:
                indi = llindex[llen.index(cor)]
                ch = 0
                for e in range(size[cor]):
                    if nseta[indi + e] == -1:
                        neg = neg + 1
                        ch = ch + 1
                asize = asize + size[cor]
            neg = neg / float(asize)
            d2 = d2 + abs(neg - rneg) * len(inBor[a][1])
            t2 = t2 + len(inBor[a][1])
        res16 = [dall / float(tall), di / float(tutto), d2 / float(t2), (di + d2) / float(tutto + t2)]
        fout2.close()

        # ME12:
        fout2 = open('ME12config' + str(ggg) + '.dat', 'w')
        fout2.write('wards KBIfraction originalfration\n')
        nseta, ml1, ml2 = GgenerateSpins(KH12, userw, Kparams, net, aarea, ppair, J12)
        llen = aarea.keys()
        llindex = []
        ssize = 0
        for w in llen:
            llindex.append(ssize)
            ssize = ssize + size[w]
        dall12 = 0.
        tall12 = 0
        di12 = 0.
        tutto12 = 0
        for cor in aarea.keys():
            if cor in Baarea.keys():
                indi = llindex[llen.index(cor)]
                rneg = aarea[cor][0][1]
                neg = 0
                for e in range(size[cor]):
                    if nseta[indi + e] == -1:
                        neg = neg + 1
                neg = neg / float(size[cor])
                dall12 = dall12 + abs(neg - rneg)
                tall12 = tall12 + 1
                di12 = di12 + abs(neg - rneg)
                tutto12 = tutto12 + 1
                fout2.write('%s %s %s\n' % (cor, neg, rneg))
            else:
                indi = llindex[llen.index(cor)]
                rneg = aarea[cor][0][1]
                neg = 0
                for e in range(size[cor]):
                    if nseta[indi + e] == -1:
                        neg = neg + 1
                neg = neg / float(size[cor])
                dall12 = dall12 + abs(neg - rneg)
                tall12 = tall12 + 1
                fout2.write('%s %s %s\n' % (cor, neg, rneg))
        d122 = 0
        t122 = 0
        for a in inBor.keys():
            neg = 0
            asize = 0
            rneg = inBor[a][0][1]
            for cor in inBor[a][1]:
                indi = llindex[llen.index(cor)]
                ch = 0
                for e in range(size[cor]):
                    if nseta[indi + e] == -1:
                        neg = neg + 1
                        ch = ch + 1
                asize = asize + size[cor]
            neg = neg / float(asize)
            d122 = d122 + abs(neg - rneg) * len(inBor[a][1])
            t122 = t122 + len(inBor[a][1])
        res12 = [dall12 / float(tall12), di12 / float(tutto12), d122 / float(t122),
                 (di12 + d122) / float(tutto12 + t122)]
        fout2.close()

        # EUref:
        fout2 = open('EUrconfig' + str(ggg) + '.dat', 'w')
        fout2.write('area(wards,Boroughs) KBIfraction originalfration\n')
        nseta, bl1, bl2 = GgenerateSpins(BKH, userw, Kparams, net, aarea, ppair, BJ)
        Btutto = 0
        Bdi = 0
        for cor in Baarea.keys():
            indi = llindex[llen.index(cor)]
            rneg = Baarea[cor][0]
            neg = 0
            for e in range(size[cor]):
                if nseta[indi + e] == -1:
                    neg = neg + 1
            neg = neg / float(size[cor])
            Bdi = Bdi + abs(neg - rneg)
            Btutto = Btutto + 1
            fout2.write('%s %s %s\n' % (cor, neg, rneg))
        Bd2 = 0
        Bt2 = 0
        for a in inBor.keys():
            neg = 0
            asize = 0
            rneg = inBor[a][0][2]
            for cor in inBor[a][1]:
                indi = llindex[llen.index(cor)]
                ch = 0
                for e in range(size[cor]):
                    if nseta[indi + e] == -1:
                        neg = neg + 1
                        ch = ch + 1
                asize = asize + size[cor]
            neg = neg / float(asize)
            Bd2 = Bd2 + abs(neg - rneg) * len(inBor[a][1])
            Bt2 = Bt2 + len(inBor[a][1])
            fout2.write('Borough %s %s %s\n' % (a, neg, rneg))
        resB = [Bdi / float(Btutto), Bd2 / float(Bt2), (Bdi + Bd2) / float(Btutto + Bt2)]
        fout2.close()
        return res16, res12, resB, avel
    else:
        return [], [], [], 100


##########################################################################
Nnumber = int(sys.argv[1])  # 获取命令参数

fh2 = open('EUwards.dat', 'r')  # 选区级别的欧盟公投数据
igot2 = fh2.readlines()
fh2.close()

Barea = {}
# Brexit data: 脱欧数据
for line in igot2:
    about = line.strip().split(' ')
    w = about[0]
    Barea[w] = []
    Barea[w].append(1. - float(about[1]))

fh = open('ME16-12_sociodemographics.dat', 'r')  # 选区级别的市长选举数据以及社会人口数据
igot = fh.readlines()
del igot[0]
fh.close()

global K
K = 5  # Blau-space dimension 布劳空间维度

area = {}
olda = 0
ind = 0

size = {}
inBor = {}
inBre = []
for line in igot:  # igot:ME16-12_sociodemographics 选区级别的市长选举数据以及社会人口数据
    about = line.strip().split(' ')
    w = about[0]  # wardCODE
    area[w] = []  # area: { wardCode:[ [ME16,ME12],[education,age,gender,[longitude,latitude],income],BoroughID ] }
    area[w].append([1. - float(about[2]), 1. - float(about[3])])  # 2:ME16 3:ME12
    size[w] = float(about[10])  # 10:size
    vec = []
    vec.append([float(about[4])])  # 4education
    vec.append([float(about[5])])  # 5age
    vec.append([float(about[6])])  # 6gender
    vec.append([float(about[7]), float(about[8])])  # 7 8 distance
    vec.append([float(about[9])])  # 9income
    area[w].append(vec)
    area[w].append(int(about[1]))  # 1BoroughID
    # list of Boroughs missing in EUreferendum data: 欧盟公投数据中缺失的自治市镇列表：
    if int(about[1]) < 33 and int(about[1]) != 7 and w not in Barea.keys():
        try:
            inBor[int(about[1])][1].append(w)
        except KeyError:
            inBor[int(about[1])] = [[], [w]]
            pass

fh.close()

# Rescale:
minsize = 40  # min population size in a ward
minw = min(size.values())
totn = 0
for w in size.keys():
    totn = totn + int((size[w] / minw) * minsize)
    size[w] = int((size[w] / minw) * minsize)

fh4 = open('ME16-12EUBoroughs.dat', 'r')  # 自治市镇级别市长选举和的欧盟公投结果
igot4 = fh4.readlines()
del igot4[0]
fh4.close()

# EUref data:
for line in igot4:
    about = line.strip().split(' ')
    a = int(about[0])
    if a in inBor.keys():
        inBor[a][0] = [1. - float(about[3]), 1. - float(about[4]), 1. - float(about[2])]  # ME2016,ME2012,EUreferendum

userw = {}
ind = 0
for w in area.keys():  # area.keys:wardCode
    for i in range(size[w]):
        userw[ind] = w
        ind = ind + 1

pairw = {}

ssum = [0.] * K
ss2 = [0.] * K
totsum = 0.

llen = area.keys()  # area.keys:wardCode
# area: { wardCode:
#                   [ [ME16,ME12],
#                     [education,
#                      age,
#                      gender,
#                      [longitude,latitude],
#                      income
#                     ],
#                     BoroughID
#                    ]
#         }
for ii in range(len(llen)):
    i = llen[ii]
    v1 = area[i][1]
    s1 = size[i]
    for jj in range(len(llen) - (ii)):
        kk = ii + jj
        k = llen[kk]
        v2 = area[k][1]
        s2 = size[k]
        if i == k:
            fff = int(s1 * (s1 - 1.) * 0.5)
        else:
            fff = s1 * s2
        pairw[(i, k)] = [0.] * (K)
        pairw[(k, i)] = [0.] * (K)

        # edu:
        absdis = abs(v1[0][0] - v2[0][0])
        pairw[(i, k)][0] = float(absdis)
        ssum[0] = ssum[0] + absdis * fff
        ss2[0] = ss2[0] + absdis * absdis * fff
        # age
        absdis = abs(v1[1][0] - v2[1][0])
        pairw[(i, k)][1] = float(absdis)
        ssum[1] = ssum[1] + absdis * fff
        ss2[1] = ss2[1] + absdis * absdis * fff
        # gender:
        absdis = abs(v1[2][0] - v2[2][0])
        pairw[(i, k)][2] = float(absdis)
        ssum[2] = ssum[2] + absdis * fff
        ss2[2] = ss2[2] + absdis * absdis * fff
        # distance:
        absdis = latlongdistance(v1[3][0], v1[3][1], v2[3][0], v2[3][1])
        pairw[(i, k)][3] = float(absdis)
        ssum[3] = ssum[3] + absdis * fff
        ss2[3] = ss2[3] + absdis * absdis * fff
        # income:
        absdis = abs(v1[4][0] - v2[4][0])
        pairw[(i, k)][4] = float(absdis)
        ssum[4] = ssum[4] + absdis * fff
        ss2[4] = ss2[4] + absdis * absdis * fff
        totsum = totsum + fff

# Blau space distance normalization:
fdisc = [0.] * K
stdv = [0.] * K
for x in range(K):
    fdisc[x] = ssum[x] / float(totsum)
    stdv[x] = sqrt((ss2[x] - 2. * ssum[x] * fdisc[x] + totsum * fdisc[x] * fdisc[x]) / float(totsum - 1))

llen = area.keys()
for ii in range(len(llen)):
    for jj in range(len(llen) - (ii)):
        kk = ii + jj
        i = llen[ii]
        k = llen[kk]
        for x in range(K):
            a = (pairw[(i, k)][x] - fdisc[x]) / (2 * stdv[x])
            pairw[(i, k)][x] = a
            pairw[(k, i)][x] = a

del fdisc
del stdv

fout = open('parameters' + str(Nnumber) + '.dat', 'w')
fout.write(
    'ID ME16distance_608w ME16distance_280w ME16distance_18Bor ME16distance_280w+18Bor ME12distance_608w ME12distance_280w ME12distance_18Bor ME12distance_280w+18Bor EUrdistance_280w EUrdistance_18Bor EUrdistance_280w+18Bor links/N theta_0 theta_edu theta_age theta_gender theta_dist theta_income ME16h_edu ME16h_age ME16h_gender ME16h_income ME12h_edu ME12h_age ME12h_gender ME12h_income EUrh_edu EUrh_age EUrh_gender EUrh_income ME16beta ME12beta EUrbeta ME16J ME12J EUrJ\n')
# random.seed(1111)
ggg = 0
start_time = time.time()
while time.time() - start_time < 84600:  # it generates spins configurations according to random model parameters for 24h
    tetas = (14., random.uniform(-7., 12.), random.uniform(-5., 12.), random.uniform(-6., 12), random.uniform(-7., 11.),
             random.uniform(-5., 12.), 0.44, random.uniform(-2.35, -0.1), random.uniform(0.3, 2.7),
             random.uniform(-1.5, -0.2), 0.44, random.uniform(-1.65, 0.15), random.uniform(0.1, 2),
             random.uniform(-1.6, -0.35), 0.44, random.uniform(-1.5, 0.15), random.uniform(-0.3, 1.2),
             random.uniform(-0.4, 0.7), random.uniform(0., 4.), random.uniform(0., 4.), random.uniform(0., 4.),
             random.uniform(0., 40.), random.uniform(0., 40.), random.uniform(0., 40.))
    params = [tetas[0], tetas[1], tetas[2], tetas[3], tetas[4], tetas[5]]
    ff = tetas[18]
    ff12 = tetas[19]
    Bff = tetas[20]
    J = tetas[18] * tetas[21]
    J12 = tetas[19] * tetas[22]
    BJ = tetas[20] * tetas[23]
    ef = [[tetas[6] * ff], [tetas[7] * ff * 0.1], [tetas[8] * ff * 10], [tetas[9] * ff * 0.1]]
    ef12 = [[tetas[10] * ff12], [tetas[11] * ff12 * 0.1], [tetas[12] * ff12 * 10], [tetas[13] * ff12 * 0.1]]
    Bef = [[tetas[14] * Bff], [tetas[15] * Bff * 0.1], [tetas[16] * Bff * 10], [tetas[17] * Bff * 0.1]]

    res16, res12, resB, avel = areadistance(K, ef, ef12, Bef, params, area, Barea, inBor, pairw, size, userw, J, J12,
                                            BJ, ggg)
    if avel != 100:
        fout.write('%s ' % (ggg))
        for a in res16:
            fout.write('%s ' % (a))
        for a in res12:
            fout.write('%s ' % (a))
        for a in resB:
            fout.write('%s ' % (a))

        fout.write('%s ' % (avel))
        for zz in range(len(tetas)):
            fout.write('%s ' % (tetas[zz]))
        fout.write('\n')
        fout.flush()
        ggg = ggg + 1
fout.close()
