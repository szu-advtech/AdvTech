import math
import matplotlib.pyplot as plt
import numpy as np

NAoi = []  # 用来装aoi


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.


def aloha_tp(lamda):  # 画fig4
    q = []  # 用来装transmission probability
    miu = np.linspace(0.01, 1, 100)
    tag = 1
    for i in miu:
        for tp in np.linspace(0, 1, 100000):  # 取样
            tag = 0
            if abs(tp * pow((1 - tp), 9) + (1 / i) * (lamda / (1 - lamda)) * tp - lamda / (1 - lamda)) < np.exp(
                    -9):
                q.append(tp)
                tag = 1
                break
        if tag == 0:
            q.append(0)
    return q


def aloha_aoi(lamda):
    # 画fig3
    NAoi_aloha = []  # 用来装aoi
    miu2 = np.linspace(0.01, 0.5, 100)
    q2 = []
    tag = 1
    print(miu2)
    for x in miu2:
        for j in np.linspace(0, 1, 1000000):  # 取样
            if abs(j * pow((1 - j), 9) + (1 / x) * (lamda / (1 - lamda)) * j - lamda / (1 - lamda)) < np.exp(
                    -8):
                q2.append(j)
                break

    # print(len(q2), end="\n")
    # print(len(miu2))

    for x, j in zip(miu2, q2):  # 每个x有j与其对应
        # print(x, j, end="\n")
        for aoi in np.linspace(0, 1200, 10000000):
            tag = 0
            if abs(((1 - lamda) / lamda) + 1 / (x * pow((1 - j), 9)) + (
                    ((1 - lamda) / pow(lamda, 2)) / (
                    ((1 - lamda) / lamda) + (1 / (x * pow((1 - j), 9))))) - aoi) < np.exp(-6):
                NAoi_aloha.append(aoi)
                tag = 1
                break
        if tag == 0:
            NAoi_aloha.append('0')
    print(len(NAoi_aloha))

    return miu2, NAoi_aloha

    # plt.plot(miu2, NAoi, label="lamda=0.05")
    # plt.xlabel("miu")
    # plt.ylabel("Network Aoi")
    # plt.show()


def csma_tp(lamda):  # L = 50
    q7 = []
    miu = np.linspace(0.05, 1, 100)
    tag = 1
    for i in miu:
        for tp in np.linspace(1, 0.001, 10000):  # q7取样
            c = pow((1 - lamda), 50)  # 其中一坨参数
            tag = 0
            if abs(c * pow((1 - tp), 9) / (1 - (1 - lamda) * pow((1 - tp), 9) - c * (1 - pow((1 - tp), 9))) + (
                    1 / i) - (1 / tp)) < np.exp(-4):
                q7.append(tp)
                tag = 1
                break
        if tag == 0:
            q7.append(0)
            print(0)

    return miu,q7

    # plt.plot(miu, q7, 'b--')
    # plt.legend(['lamda=0.0019'], loc=2)
    # plt.xlabel("miu")
    # plt.ylabel("transport probability")
    # plt.show()


if __name__ == '__main__':
    miu1, NAoI_1 = aloha_aoi(0.5)  # 分析模型的部分
    print(miu1)
    # miu2, NAoI_2 = aloha_aoi(0.05)
    plt.plot(miu1, NAoI_1, 'darkorange', label='Analytical Model for λ=0.5')

    # miu = np.linspace(0.01, 1, 100)
    # q = aloha_tp(0.5)
    # plt.plot(miu, q, label="lamda=0.05")
    # plt.xlabel("miu")
    # plt.ylabel("transport probability")
    # plt.show()

    # aloha_aoi(0.5)
    # csma_tp(0.05)
