import numpy as np
import sys
from math import *

# 2016 MEwards
fh = open('OriginalData/gla-elections-votes-all-2016_wards.csv', 'r')
# https://data.london.gov.uk/dataset/london-elections-results-2016-wards-boroughs-constituency
igot = fh.readlines()
del igot[0]
del igot[0]
del igot[0]
del igot[0]
fh.close()

lasta = 'Bexley'
area = 0
ward = 0
darea = {}
dward = {}

fh4 = open('OriginalData/2011censuswardscoord.csv', 'r')
# from infuse_ward_lyr_2011.shp computed centroids using geopandas
igot4 = fh4.readlines()
centro = {}
for line in igot4:
    about = line.strip().split(',')
    centro[about[0]] = [about[1], about[2]]
fh4.close()

darea[lasta] = area

moredw = {}
moredw['E05009367'] = 'E05000231'
moredw['E05009368'] = 'E05000232'
moredw['E05009369'] = 'E05000234'
moredw['E05009370'] = 'E05000235'
moredw['E05009371'] = 'E05000236'
moredw['E05009372'] = 'E05000237'
moredw['E05009373'] = 'E05000238'
moredw['E05009374'] = 'E05000239'
moredw['E05009378'] = 'E05000241'
moredw['E05009379'] = 'E05000242'
moredw['E05009382'] = 'E05000246'
moredw['E05009385'] = 'E05000248'
moredw['E05009317'] = 'E05000573'
moredw['E05009318'] = 'E05000575'
moredw['E05009319'] = 'E05000576'
moredw['E05009320'] = 'E05000577'
moredw['E05009326'] = 'E05000580'
moredw['E05009332'] = 'E05000586'
moredw['E05009333'] = 'E05000587'
moredw['E05009329'] = 'E05000584'
moredw['E05009330'] = 'E05000585'
moredw['E05009334'] = 'E05000584'
moredw['E05009335'] = 'E05000588'
moredw['E05009336'] = 'E05000589'
moredw['E05009388'] = 'E05000382'
moredw['E05009389'] = 'E05000383'
moredw['E05009390'] = 'E05000384'
moredw['E05009392'] = 'E05000385'
moredw['E05009393'] = 'E05000386'
moredw['E05009395'] = 'E05000388'
moredw['E05009396'] = 'E05000389'
moredw['E05009397'] = 'E05000391'
moredw['E05009398'] = 'E05000392'
moredw['E05009399'] = 'E05000393'
moredw['E05009400'] = 'E05000394'
moredw['E05009401'] = 'E05000395'
moredw['E05009402'] = 'E05000396'
moredw['E05009403'] = 'E05000397'
moredw['E05009405'] = 'E05000399'
defdic = {}
forget = 0.
more = 0
for line in igot:
    about = line.strip().split(',')
    saq = float(about[11])
    cons = float(about[12])
    re = float(about[11]) / (saq + cons)
    if about[0] != lasta:
        lasta = about[0]
        area = area + 1
        darea[about[0]] = area
    forget = forget + (float(about[23]) - (saq + cons)) / float(about[23])
    if area in [7, 9, 24, 31]:
        try:
            defdic[moredw[about[4]]] = [area, re]
            dward[moredw[about[4]]] = ward
            more = more + 1
        except KeyError:
            defdic[about[4]] = [area, re]
            dward[about[4]] = ward
    else:
        defdic[about[4]] = [area, re]
        dward[about[4]] = ward
    ward = ward + 1

# 2012 ME ward data:
fh2 = open('OriginalData/Lower_Layer_Super_Output_Area__2001__to_Ward__2010__Lookup_in_England_and_Wales.csv', 'r')
# https://geoportal.statistics.gov.uk/datasets/lower-layer-super-output-area-2001-to-ward-2010-lookup-in-england-and-wales/data
igot2 = fh2.readlines()
del igot2[0]
trans = {}
for line in igot2:
    about = line.strip().split(',')
    trans[about[3]] = about[2]
fh2.close()

fh2 = open('OriginalData/gla-elections-votes-ward-2012_wards.csv', 'r')
# https://data.london.gov.uk/dataset/london-elections-results-2012-wards-boroughs-constituency
igot2 = fh2.readlines()
del igot2[:11]

res12 = {}

for line in igot2:
    about = line.strip().split(',')
    saq = float(about[13])
    cons = float(about[11])
    re = saq / (saq + cons)
    try:
        defdic[trans[about[0]]].append(re)
    except KeyError:
        pass
fh2.close()

# Income:
fh2 = open('OriginalData/ward-profiles-excel-version.csv', 'r')
# https://data.london.gov.uk/dataset/ward-profiles-and-atlas
igot2 = fh2.readlines()
del igot2[0]

inc = {}

for line in igot2:
    about = line.strip().split(',')
    inc[about[2]] = float(about[32]) / 1000.  # income(£1000)

fh2.close()

# 2011Census Varaibles:
fh = open('OriginalData/2017121816246549_AGE_HIQUAL_UNIT/Data_AGE_HIQUAL_UNIT.csv', 'r')
igot = fh.readlines()
del igot[0]
fh.close()

tot = 0
nw = 0
for line in igot:
    about = line.strip().split(',')
    if about[1] in defdic.keys():
        loc = []
        data = []
        for i in range(5):
            n = int(about[len(about) - (6 - i)])
            data = data + [i] * n
            tot = tot + n
            loc.append(float(about[len(about) - (6 - i)]))
        nw = nw + 1
        sis = sum(loc)
        amed = 0.
        for i in range(5):
            amed = amed + i * loc[i] / float(sis)
        defdic[about[1]].append(amed)

fh = open('OriginalData/1b12018115151351415_AGE_UNIT/Data_AGE_UNIT.csv', 'r')
igot = fh.readlines()
igori = igot[1]
oriab = igori.strip().split(',')
age = []
for i in range(87):
    try:
        a = oriab[len(oriab) - (88 - i)]
        b = float(a.replace('Age : Age ', '').replace(' - Unit : Persons', ''))
        age.append(b)
    except ValueError:
        age.append(100.)
del igot[0]
del igot[1]
fh.close()

tot = 0
nw = 0
for line in igot:
    about = line.strip().split(',')
    if about[1] in defdic.keys():
        data = []
        sis = 0.
        amed = 0.
        for i in range(87):
            if age[i] > 18:
                n = int(float(about[len(about) - (88 - i)]))
                sis = sis + n
                amed = amed + age[i] * n
                data = data + [age[i]] * n
                tot = tot + n
        nw = nw + 1
        amed = amed / float(sis)
        defdic[about[1]].append(amed)

fh = open('OriginalData/201812131062425_AGE_SEX_UNIT/Data_AGE_SEX_UNIT.csv', 'r')
igot = fh.readlines()
igori = igot[1]
oriab = igori.strip().split(',')
gen = []
gen2 = []
for i in range(40):
    aa = oriab[len(oriab) - (41 - i)]
    gen2.append(aa)
    a = aa.strip().split(' ')
    gen.append(a.count('Females'))
fh.close()

tot = 0
nw = 0
size = []
for line in igot:
    about = line.strip().split(',')
    if about[1] in defdic.keys():
        data = []
        sis = 0.
        amed = 0.
        for i in range(40):
            n = int(about[len(about) - (41 - i)])
            sis = sis + n
            amed = amed + gen[i] * n
            data = data + [gen[i]] * n
            tot = tot + n
        nw = nw + 1
        amed = amed / float(sis)
        defdic[about[1]].append(amed)
        defdic[about[1]].append(centro[about[1]][0])
        defdic[about[1]].append(centro[about[1]][1])
        defdic[about[1]].append(inc[about[1]])
        defdic[about[1]].append(sis)  # size according to census data

# outcome file ward level(608) 2012 2016 MEs + sociodemographic data:
fout = open('ME16-12_sociodemographics.dat', 'w')
fout.write('0wardCODE 1BoroughID 2ME16 3ME12 4education 5age 6gender 7longitude 8latitude 9income(£1000) 10size\n')
for e in defdic.keys():
    if len(defdic[e]) > 5:
        fout.write('%s ' % e)
        for i in range(len(defdic[e])):
            fout.write('%s ' % defdic[e][i])
        fout.write('\n')
fout.close()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~EU Referendum data and Borough data
fh = open('OriginalData/ward-results.csv', 'r')
# https://3859gp38qzh51h504x6gvv0o-wpengine.netdna-ssl.com/files/2017/02/ward-results.xlsx
# outcome file ward level EUref data:
fout = open('EUwards.dat', 'w')
igot = fh.readlines()
for line in igot:
    about = line.strip().split(',')
    if about[0] in defdic.keys():
        try:
            a = float(about[6]) / 100.
            fout.write('%s %s\n' % (about[0], a))
        except ValueError:
            pass
fh.close()
fout.close()

GLBoroughs = ['E09000002', 'E09000003', 'E09000004', 'E09000005', 'E09000006', 'E09000007', 'E09000008', 'E09000009',
              'E09000010',
              'E09000011', 'E09000012', 'E09000013', 'E09000014', 'E09000015', 'E09000016', 'E09000017', 'E09000018',
              'E09000019', 'E09000020',
              'E09000021', 'E09000022', 'E09000023', 'E09000024', 'E09000025', 'E09000026', 'E09000027', 'E09000028',
              'E09000029', 'E09000030',
              'E09000031', 'E09000032', 'E09000033']
Bou = {}

# EUref Borough level data:
fh = open('OriginalData/EU-referendum-result-data.csv', 'r')
# https://data.london.gov.uk/dataset/eu-referendum-results EU-referendum-result-data.csv
igot = fh.readlines()
del igot[0]
for line in igot:
    about = line.strip().split(',')
    if about[3] in GLBoroughs:
        Bou[about[3]] = [float(about[11]) / (float(about[11]) + float(about[12]))]
fh.close()

# Borough level ME outcomes:
fh = open('OriginalData/gla-elections-2000-2016.csv', 'r')
# Borough level all together: https://data.london.gov.uk/dataset/london-elections-results-2016-wards-boroughs-constituency
# gla-elections-2000-2016.csv
igot = fh.readlines()
del igot[0]
ands = {}
for line in igot:
    about = line.strip().split(',')
    if about[0] in GLBoroughs:
        Bou[about[0]].append(float(about[21]) / (float(about[21]) + float(about[17])))  # 2016
        Bou[about[0]].append(float(about[20]) / (float(about[20]) + float(about[16])))  # 2016
        ands[about[0]] = about[1]
        if about[1] == 'Barking and Dagenham':
            ands[about[0]] = 'Barking & Dagenham'
        if about[1] == 'Hammersmith and Fulham':
            ands[about[0]] = 'Hammersmith & Fulham'
        if about[1] == 'Kensington and Chelsea':
            ands[about[0]] = 'Kensington & Chelsea'

# outcome file Borough lebel MEs:
fout = open('ME16-12EUBoroughs.dat', 'w')
fout.write('ID,Code,EUreferendum,ME2016,ME2012\n')
for e in Bou.keys():
    fout.write('%s %s %s %s %s\n' % (darea[ands[e]], e, Bou[e][0], Bou[e][1], Bou[e][2]))
fout.close()
