import csv
import math

import numpy as np
import pandas


def read_csv(path):
    with open(path, 'r', newline='') as csv_file:
        reader = csv.reader(csv_file)
        cd_data = []
        cd_data += [[float(i) for i in row] for row in reader]
        return cd_data


def save_csv(data, path):
    pandas.DataFrame(data=data).to_csv("./my/" + path, index=False, header=False)


dss = read_csv('dss.csv')


def getDS(di, Dj):
    return max([dss[di][j] for j in Dj])


c_d = read_csv('c_d.csv')

CcontainsD = []
maxdss = []

for i in c_d:
    CcontainsD.append([k for k, j in enumerate(i) if j])

cfs = [[0 for i in range(585)] for j in range(585)]
for i in range(585):
    for j in range(585):
        cfs[i][j] = (sum([getDS(k, CcontainsD[j]) for k in CcontainsD[i]]) + sum(
            [getDS(k, CcontainsD[i]) for k in CcontainsD[j]])) / \
                    len(CcontainsD[i] + CcontainsD[j])
save_csv(cfs, "cfs.csv")

cgs = [[0 for i in range(585)] for j in range(585)]
p = 585 / (sum(sum(i) for i in c_d))
for i in range(585):
    for j in range(585):
        cgs[i][j] = math.exp(-p * sum((e - f) ** 2 for e, f in zip(c_d[i], c_d[j])))
save_csv(cgs, "cgs.csv")

dgs = [[0 for i in range(88)] for j in range(88)]
d_c = np.array(c_d).transpose()
p = 88 / (sum(sum(i) for i in d_c))
for i in range(88):
    for j in range(88):
        dgs[i][j] = math.exp(-p * sum((e - f) ** 2 for e, f in zip(d_c[i], d_c[j])))
save_csv(dgs, "dgs.csv")

c_c = [[cfs[i][j] if cfs[i][j] else cgs[i][j] for i in range(585)] for j in range(585)]
save_csv(c_c, "c_c.csv")

d_d = [[dss[i][j] if dss[i][j] else dgs[i][j] for i in range(88)] for j in range(88)]
save_csv(d_d, "d_d.csv")
