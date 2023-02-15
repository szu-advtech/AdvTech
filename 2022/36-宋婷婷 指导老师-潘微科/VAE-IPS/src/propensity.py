import numpy as np

def create_propensity():
    num_users = 15400
    num_items = 1000

    # Ui_num: 与物品i交互过的用户总数
    Ui_num = np.zeros(num_items)

    # Iu_num: 用户u交互过的物品总数
    Iu_num = np.zeros(num_users)

    # 读取数据
    f = open('../data/train.txt')
    for line in f.readlines():
        u, i, r = line.split(',')
        u = int(u)
        i = int(i)

        Ui_num[i] += 1
        Iu_num[u] += 1
    f.close()

    # 计算用户u的倾向得分
    propensity_u = (Iu_num / max(Iu_num)) ** 0.5

    # 计算物品i的倾向得分theta_i
    propensity_i = (Ui_num / max(Ui_num)) ** 0.5

    # 计算最终倾向得分
    propensity_u = np.expand_dims(propensity_u, 1)
    propensity_i = np.expand_dims(propensity_i, 1)
    propensity = np.dot(propensity_u, propensity_i.T)

    return propensity


if __name__ == '__main__':
    create_propensity()
