import numpy as np
import  pandas  as pd

print("----读取，数据会存在列表里面----")
df = pd.read_csv('../NNDR_feature_vectors/15scenes.dat', sep="\t")
print("pd.read_csv：", df)

# 标签
label = df.iloc[:, 0]
print("读取label：", label)
print("读取label的shape:", label.shape[0])

# feature
vector = df.iloc[:, 1 : df.shape[1] - 1]
print("读取vector：", vector)
print("读取vector的shape:", vector.shape)

