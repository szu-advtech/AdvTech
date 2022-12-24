# coding=utf-8
from datasketch import MinHash, MinHashLSH
import sys
import pandas as pd
import numpy as np
from main import learn_model

# 为数据帧中的每一列创建一个minhash。
min_hashes = []
data = pd.read_csv(sys.argv[1], dtype=str)
output_truth = np.zeros((len(data.columns.values), len(data.columns.values)))
for col in data.columns.values:
    model = learn_model(data[col].tolist())['model']
    min_hash = MinHash(num_perm=128)
    num_updates = 0
    while num_updates < 128:
        for s in model.generate_mh_strings():
            min_hash.update(s.encode('utf8'))
            num_updates += 1
            if num_updates == 128:
                break
        else:
            break
    min_hashes.append(min_hash)

# 创建LSH对象，并将min_hash插入到LSH对象中。
lsh = MinHashLSH(threshold=0.5, num_perm=128)
i = 0
for min_hash in min_hashes:
    lsh.insert(str(i), min_hash)
    i += 1

# 查找每一列与其他列的匹配。
j = 0
for mh in min_hashes:
    matches = lsh.query(mh)
    for match in matches:
        output_truth[j][int(match)] = 1
        output_truth[int(match)][j] = 1
    j += 1

print output_truth
