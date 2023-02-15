import random
import matplotlib.pyplot as plt
import numpy as np

import analy_model

N = 10  # 节点数
TSLOTS = 200000  # 时间片总数
miu = np.linspace(0.01, 0.5, 66)
success = []


def sim_aloha_aoi(lamda):
    n_aoi = []
    for j in miu:  # 以概率j发送包
        trans_wait = []  # 每个分槽生成的但未传输的
        time_r = []  # 每个节点发送成功的最新时间
        for _ in range(N):
            trans_wait.append(0)
        # print(trans_wait)

        for _ in range(N):
            time_r.append(0)
        trans_d = 0  # 记录概率为可以发送出去的包（不管碰撞）

        for time in range(TSLOTS):
            for i in range(N):
                if trans_wait[i] == 0:
                    r = random.randint(0, 100)
                    if r < lamda * 100:  # 表示生成新的包，没包的要按概率生成包
                        trans_wait[i] = 1

            # if flag == 1:
            #     flag = 0
            #     continue
            # print(trans_wait)
            transmitted = []  # 标记该轮发送的包
            k = 0  # 记录每轮概率为将要发送的包
            for i in trans_wait:
                if i == 1:
                    a = random.randint(0, 1000)
                    if a < j * 1000:  # j是发送的概率
                        # print(a)
                        transmitted.append(1)
                        k = k + 1
                        trans_d = trans_d + 1
                    else:
                        transmitted.append(0)
                else:
                    transmitted.append(0)

            if k == 1:  # 代表该轮发送成功
                for i in range(N):  # 看是哪个节点发送成功
                    if transmitted[i] == 1:
                        # print(time)
                        time_r[i] = time  # 更新记录i节点最新发送成功包的时隙
                        trans_wait[i] = 0  # 发送出去了 置为0

        # print(trans_wait)

        aoi = 0
        live_node = 0  # 发送过数据包的节点
        for x in range(10):
            if time_r[x] > 0:
                n_time = TSLOTS - time_r[x]
                aoi += n_time
                live_node += 1

        if live_node != 0:
            aoi /= live_node
            print(aoi)
            n_aoi.append(aoi)
        else:
            print("error")

    return n_aoi


if __name__ == "__main__":
    n_aoi_1 = []
    n_aoi_2 = []
    n = len(miu)
    times = 10  # 循环次数
    for _ in range(n):
        n_aoi_1.append(0)
        n_aoi_2.append(0)
    for _ in range(times):
        aoi_1 = sim_aloha_aoi(0.5)
        print(aoi_1)
        aoi_2 = sim_aloha_aoi(0.05)
        for i in range(n):
            n_aoi_1[i] += aoi_1[i]
            n_aoi_2[i] += aoi_2[i]
    for j in range(n):
        n_aoi_1[j] /= times
        n_aoi_2[j] /= times

    miu1, NAoI_1 = analy_model.aloha_aoi(0.5)   # 分析模型的部分
    print(miu1)
    miu2, NAoI_2 = analy_model.aloha_aoi(0.05)
    plt.plot(miu1, NAoI_1, 'b--', label='Analytical Model for λ=0.5')
    plt.plot(miu2, NAoI_2, 'r--', label='Analytical Model for λ=0.05')
    plt.plot(miu, n_aoi_1,'darkorange', label='Simulation for λ=0.5')
    plt.plot(miu, n_aoi_2, label='Simulation for λ=0.05')
    plt.xlabel("Conditional transmission probability μ")
    plt.legend(loc="best")
    plt.ylabel("Network AoI")
    plt.show()
