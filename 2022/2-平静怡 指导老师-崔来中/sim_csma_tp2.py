import random
import matplotlib.pyplot as plt
import numpy as np

N = 10  # 节点数
TSLOTS = 100000  # 时间片总数
n_aoi = []


def sim_csma_tp(lamda):
    miu = np.linspace(0.01, 1, 100)
    success = []
    for j in miu:  # 以概率j发送包
        collision = 0
        time_r = []
        count = 0  # 用来标记以概率j发送包能发送成功的包的个数
        trans_wait = []  # 每个分槽生成的但未传输的
        trans_d = 0  # 记录概率为可以发送出去的包（不管碰撞）
        for _ in range(N):
            trans_wait.append(0)

        kick = 0  # 代表L

        for time in range(0, TSLOTS):
            if kick == 0:
                collision = 0
            # print(trans_wait)
            # print("time is {}".format(time))
            for i in range(N):
                if trans_wait[i] == 0:
                    r = random.randint(0, 10000)
                    if r < lamda * 10000:  # 表示生成新的包，没包的要按概率生成包
                        trans_wait[i] = 1
            transmitted = []  # 标记该轮发送的包
            k = 0  # 记录每轮概率为将要发送的包

            # for i_2 in trans_wait:
            #     if i_2 == 1:
            #         a = random.randint(0, 100)  # 有概率发送的
            #         if a < j * 100:  # 这里发送的概率
            #             transmitted.append(1)
            #             k = k + 1
            #             trans_d = trans_d + 1
            #         else:
            #             transmitted.append(0)
            #     else:
            #         transmitted.append(0)

            if collision == 0:
                for i_2 in trans_wait:
                    if i_2 == 1:
                        a = random.randint(0, 100)  # 有概率发送的
                        if a < j * 100:  # 这里发送的概率
                            transmitted.append(1)
                            k = k + 1
                            trans_d = trans_d + 1
                        else:
                            transmitted.append(0)
                    else:
                        transmitted.append(0)
            elif collision == 1:
                for i_3 in trans_wait:
                    if i_3 == 1:
                        a = random.randint(0, 100)  # 有概率发送的
                        if a < j * 100:  # 这里发送的概率
                            trans_d = trans_d + 1

            # print(len(transmitted))
            # print(transmitted)
            if k == 1:  # 代表该轮发送成功
                kick = 50  # L为50
                collision = 1
                count = count + 1
                for i in range(10):  # 看是哪个节点发送成功
                    if transmitted[i] == 1:
                        trans_wait[i] = 0  # 发送出去了，置为0
                        # print(time)
                        # time_r[i] = time  # 更新记录i节点最新发送成功包的时隙

            if k > 1:
                collision = 1
                kick = 50

            if kick > 0:
                kick -= 1

        # print(trans_wait)
        tp = trans_d / (TSLOTS * N)
        print(tp)
        success.append(tp)
    return miu,success


if __name__ == "__main__":
    miu,success_1 = sim_csma_tp(0.0013)
    miu1,success_2 = sim_csma_tp(0.0019)
    plt.plot(miu, success_1, 'darkorange', label='Simulation for λ=0.0013')
    plt.plot(miu, success_2, label='Simulation for λ=0.0019')
    plt.legend(loc='best')
    plt.xlabel("Conditional transmission probability μ")
    plt.ylabel("transport probability")
    plt.show()
