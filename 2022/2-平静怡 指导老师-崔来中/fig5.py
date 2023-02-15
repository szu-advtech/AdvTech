import numpy as np
from matplotlib import pyplot as plt
import gra
import sim_aloha_tp
import sim_aloha_aoi
import analy_model
from sim_csma_aoi import sim_csma_aoi

if __name__ == "__main__":
    miu1,csma_aoi_1 = gra.csma_aoi(0.5)
    miu2, csma_aoi_2 = gra.csma_aoi(0.05)
    plt.plot(miu1, csma_aoi_1, 'b--', label='Analytical Model for λ=0.5')
    plt.plot(miu1, csma_aoi_2, 'r--', label='Analytical Model for λ=0.05')

    miu = np.linspace(0.01, 0.3, 60)
    n_aoi_1 = []
    n_aoi_2 = []
    n = len(miu)
    times = 20  # 循环次数
    for _ in range(n):
        n_aoi_1.append(0)
        n_aoi_2.append(0)
    for _ in range(times):
        aoi_1 = sim_csma_aoi(0.5)
        # print(aoi_1)
        aoi_2 = sim_csma_aoi(0.05)
        for i in range(n):
            n_aoi_1[i] += aoi_1[i]
            n_aoi_2[i] += aoi_2[i]
    for b in range(n):
        n_aoi_1[b] /= times
        n_aoi_2[b] /= times
    aoi_1 = sim_csma_aoi(0.5)
    aoi_2 = sim_csma_aoi(0.05)
    plt.plot(miu, n_aoi_1, 'darkorange', label='Simulation for λ=0.5')
    plt.plot(miu, n_aoi_2, label='Simulation for λ=0.05')
    plt.legend(loc='best')
    plt.xlabel("Conditional transmission probability μ")
    plt.ylabel("Network AoI")
    plt.show()