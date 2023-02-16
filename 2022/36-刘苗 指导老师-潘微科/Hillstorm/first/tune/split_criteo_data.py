import numpy as np
import pandas as pd
from pandas import DataFrame
if __name__ == "__main__":
    data = pd.read_csv('../deal_Hillstrom.csv')
    data_random_gene = DataFrame.sample(data, frac=1)
    split_point = int(len(data_random_gene) * 0.8)
    train_data = data_random_gene.iloc[:split_point, :]
    test_data = data_random_gene.iloc[split_point:, :]
    #保存到csv文件中
    train_data.to_csv('../train_Hillstrom.csv',sep=',',header=True,index=False)
    test_data.to_csv('../test_Hillstrom.csv',sep=',',header=True,index=False)
    # takes a contrastive pair
    g1_g4 = []
    g2_g3 = []
    for index, step in train_data.iterrows():
        if np.logical_xor(int(step["8"]), int(step["9"])) == 0:  # 异或T,Label
            g1_g4.append(step)
        else:
            g2_g3.append(step)
    # 乱序俩个集合
    np.random.shuffle(g1_g4)
    np.random.shuffle(g2_g3)

    # 随机选择x1,x2构建偏序对，并保存在数组中，同时还有超参数y_diff
    np.random.shuffle(g1_g4)
    np.random.shuffle(g2_g3)
    print(len(g1_g4))
    print(len(g2_g3))
    print(len(train_data)==(len(g1_g4)+len(g2_g3)))
    # 随机选择x1,x2构建偏序对，并保存在数组中，同时还有超参数y_diff
    contrastive_pair = []

    # 构建多少个这样的偏序对
    for iter in range(len(g2_g3)):
        if int(np.round(np.random.random())) == 1:
            if iter >= len(g1_g4):
                x1 = g1_g4[np.random.randint(0, len(g1_g4) - 1)]
            else:
                x1 = g1_g4[iter]
            x2 = g2_g3[iter]

        else:
            if iter >= len(g1_g4):
                x2 = g1_g4[np.random.randint(0, len(g1_g4) - 1)]
            else:
                x2 = g1_g4[iter]
            x1 = g2_g3[iter]

        contrastive_pair.append(
            [x1[0], x1[1], x1[2], x1[3], x1[4], x1[5], x1[6], x1[7],int(x1[8]),int(x1[9]),
             x2[0], x2[1], x2[2], x2[3], x2[4], x2[5], x2[6], x2[7],int(x2[8]),int(x2[9]),
              ])
    #将偏序对变为DataFrame数据
    contrastive_pair = pd.DataFrame(contrastive_pair)
    contrastive_pair.to_csv('../contrastive_pair_Hillstrom.csv', sep=',', header=True, index=False)



