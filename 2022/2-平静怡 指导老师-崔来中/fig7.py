import numpy as np
from matplotlib import pyplot as plt
import analy_model
import sim_csma_tp2

if __name__ == "__main__":
    miu1, csma_tp1 = analy_model.csma_tp(0.0013)
    miu2, csma_tp2 = analy_model.csma_tp(0.0019)
    miu = np.linspace(0.01, 1, 100)

    times = 10
    sim_csma_tp_13 = []  # 去sim_csma_tp2里看，结果长度为100
    sim_csma_tp_19 = []
    for _ in range(100):
        sim_csma_tp_13.append(0)
        sim_csma_tp_19.append(0)

    for _ in range(times):
        miu3, sim_csma_tp13 = sim_csma_tp2.sim_csma_tp(0.0013)
        miu4, sim_csma_tp19 = sim_csma_tp2.sim_csma_tp(0.0019)
        for i in range(100):
            sim_csma_tp_13[i] += sim_csma_tp13[i]
            sim_csma_tp_19[i] += sim_csma_tp19[i]

    for j in range(100):
        sim_csma_tp_13[j] /= times
        sim_csma_tp_19[j] /= times

    plt.plot(miu, sim_csma_tp_13, 'darkorange', label='Simulation for λ=0.0013')
    plt.plot(miu1, csma_tp1, 'b--', label='Analytical Model for λ=0.0013')
    plt.plot(miu, sim_csma_tp_19, label='Simulation for λ=0.0019')
    plt.plot(miu1, csma_tp2, 'r--', label='Analytical Model for λ=0.0019')
    plt.legend(loc='best')
    plt.xlabel("Conditional transmission probability μ")
    plt.ylabel("Transport probability")
    plt.show()
