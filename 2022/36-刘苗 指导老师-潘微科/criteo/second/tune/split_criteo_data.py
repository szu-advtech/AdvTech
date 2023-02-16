import numpy as np
import pandas as pd
from pandas import DataFrame
if __name__ == "__main__":
    data = pd.read_csv('../criteo-uplift-v2.1.csv')
    data_random_gene = DataFrame.sample(data, frac=0.01)
    # print(len(data_random_gene))
    split_point = int(len(data_random_gene) * 0.7)
    # print("split_point",split_point)
    train_data = data_random_gene.iloc[:split_point, :]
    test_data = data_random_gene.iloc[split_point:, :]
    # print(len(train_data))
    # print(len(test_data))
    #data.npy
    train_data.to_csv('../train_criteo.csv', sep=',', header=True, index=False)
    test_data.to_csv('../test_criteo.csv', sep=',', header=True, index=False)

    # takes a contrastive pair
    g1_g4 = []
    g2_g3 = []
    for index, step in train_data.iterrows():
        if np.logical_xor(int(step["treatment"]), int(step["conversion"])) == 0:  # 异或T,Label
            g1_g4.append(step)
        else:
            g2_g3.append(step)
    # 乱序俩个集合
    np.random.shuffle(g1_g4)
    np.random.shuffle(g2_g3)
    print(len(g1_g4))
    print(len(g2_g3))
    # 随机选择x1,x2构建偏序对，并保存在数组中，同时还有超参数y_diff
    contrastive_pair = []
    # 构建多少个这样的偏序对
    for iter in range(len(g2_g3)):
        if int(np.round(np.random.random())) == 1:
            if iter >= len(g1_g4):
                x1=g1_g4[np.random.randint(0, len(g1_g4) - 1)]
            else: x1 = g1_g4[iter]
            x2 = g2_g3[iter]
        else:
            if iter >= len(g1_g4):
                x2=g1_g4[np.random.randint(0, len(g1_g4) - 1)]
            else:  x2 = g1_g4[iter]
            x1 = g2_g3[iter]
        contrastive_pair.append(
            [x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], x1[6], x1[7], x1[8], x1[9], x1[10], x1[11], int(x1[12]),int(x1[13]),
             x2[0], x2[1], x2[2], x2[3], x2[4], x2[5], x2[6], x2[7], x2[8], x2[9], x2[10], x2[11], int(x2[12]),int(x2[13]),
             ])
    #将偏序对变为DataFrame数据
    contrastive_pair = pd.DataFrame(contrastive_pair)
    contrastive_pair.to_csv('../contrastive_pair_criteo.csv', sep=',', header=True, index=False)



