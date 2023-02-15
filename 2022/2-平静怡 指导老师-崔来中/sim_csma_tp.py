import random
from math import ceil, floor

import matplotlib.pyplot as plt
import numpy as np

N = 10  # 节点数
TSLOTS = 10000  # 时间片总数
miu = np.linspace(0.01, 1, 100)
n_aoi = []
L = 50
cw_min = 1
result = []


def sim_csma(lamda):
    print("begin")
    for j in miu:  # 以概率j发送包
        back_off = []
        # cw_val = []  # 存放每个节点的窗口值
        trans_wait = []
        collision = 0  # 记录目前是否有节点发送，0为当前空闲
        trans_d = 0  # 记录概率为可以发送出去的包（不管碰撞）
        transmitted = []  # 记录当前在传输的
        cw_val = int(2 / j - 1)
        # print(j,cw_val)
        print("miu is:{},cw_val:{}".format(j, cw_val))
        for _ in range(N):
            # cw_val.append(cw_min)  # 初始化各节点的窗口
            back_off.append(0)
            trans_wait.append(0)
            transmitted.append(0)

        # print("********************")
        # print("beginning backoff is {}".format(back_off))

        tick = 0

        for time in range(TSLOTS):
            for r in range(N):
                transmitted[r] = 0

            for pack in range(N):
                if trans_wait[pack] == 0:
                    r = random.randint(0, 100)
                    if r < lamda * 100:  # 表示每个时隙都有机会生成新的包，没包的要按概率生成包
                        trans_wait[pack] = 1
                        back_off[pack] = random.randint(0, cw_val)  # 生成新包的节点的backoff
                        # back_off[pack] = cw_val  # 生成新包的节点的backoff

            if tick == 0:  # 信道空闲
                collision = 0
                # for td in range(N):
                #     if transmitted[td] == 1:
                #         transmitted[td] = 0  # 当前传输列表归零
                # back_off[td] = 0
            # print("********************")
            # print("now the trans_wait is :{}".format(trans_wait))
            # print("now backoff{},cw is:{}".format(back_off, cw_val))

            # transmitted = np.zeros(10, int)  # 标记该轮发送的包
            k = 0  # 记录每轮概率为将要发送的包

            for i in range(N):
                # print("trans_wait:{}".format(trans_wait))
                if trans_wait[i] == 1:  # 有包
                    if collision == 0 and back_off[i] == 0:  # 目前无冲突且backoff倒数为0
                        k += 1
                        trans_d += 1
                        transmitted[i] = 1  # 标记该节点发送
                        # a = random.randint(0, 100)  # 有概率发送的
                        # if a < j * 100:  # j为发送的概率
                        #     k += 1
                        #     trans_d += 1
                        #     transmitted[i] = 1  # 标记该节点发送

                        continue

                    elif collision == 0 and back_off[i] != 0:  # 目前无冲突 但当前节点backoff还没倒数到0
                        back_off[i] -= 1  # backoff时钟在空闲时-1
                        # a = random.randint(0, 100)  # 有概率发送的
                        # if a < j * 100:  # j为发送的概率（但发送不成功 因为有帧在发送）
                        #     trans_d += 1
                        continue

                    elif collision == 1 and back_off[i] == 0:  # 当前有帧在发送,并且当前节点要发送了（backoff成0）
                        trans_d += 1
                        back_off[i] = (cw_val)  # 重新调整了backoff
                        # back_off[pack] = random.randint(0, cw_val)
                        # a = random.randint(0, 100)  # 有概率发送的
                        # if a < j * 100:  # j为发送的概率
                        #     trans_d += 1
                        #     back_off[i] = cw_val # 重新调整了backoff

                        # a = random.randint(0, 100)  # 有概率发送的
                        # if a < j * 100:  # j为发送的概率（但发送不成功 因为有帧在发送）
                        #     trans_d += 1
                        #     cw_val[i] *= 2  # 加倍后退选择的时间槽数
                        #     if cw_val[i] > 1024:
                        #         cw_val[i] = 1024  # 最大不超过1024

                        continue

                    # elif collision == 1 and back_off[i] != 0:
                    #     back_off[i] -= 1
                    #     continue

            if k == 1:  # 可成功发送
                for a in range(N):
                    if transmitted[a] == 1:
                        # print("发送成功一个{}".format(i))
                        collision = 1  # 表示现在有节点在发送，不空闲
                        tick = 50  # 该包发送时间d
                        trans_wait[a] = 0  # 发送出去了

            if k > 1:  # 发送不成功，碰撞
                # collision = 1
                # tick = 50
                for b in range(N):
                    if transmitted[b] == 1:
                        # back_off[pack] = random.randint(0, cw_val)
                        back_off[b] = cw_val
                        # transmitted[b] = 0

            if tick > 0:
                tick -= 1

        # print(trans_d)

        tp = trans_d / (TSLOTS * N)
        # print(tp)
        result.append(tp)

    plt.plot(miu, result)
    plt.show()


if __name__ == "__main__":
    sim_csma(0.5)
