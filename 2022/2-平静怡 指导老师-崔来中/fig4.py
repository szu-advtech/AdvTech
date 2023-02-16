import numpy as np
from matplotlib import pyplot as plt

import sim_aloha_tp
import sim_aloha_aoi
import analy_model

if __name__ == "__main__":
    miu = np.linspace(0.01, 1, 100)
    q_1 = analy_model.aloha_tp(0.5)
    q_2 = analy_model.aloha_tp(0.05)
    success_1 = sim_aloha_tp.sim_aloha_tp(0.5)
    success_2 = sim_aloha_tp.sim_aloha_tp(0.05)

    plt.plot(miu, success_1, 'darkorange', label='Simulation for λ=0.5')
    plt.plot(miu, q_1, 'b--', label="Analytical Model for λ=0.5")
    plt.plot(miu, success_2, label='Simulation for λ=0.05')
    plt.plot(miu, q_2, 'r--', label="Analytical Model for λ=0.05")
    plt.legend(loc='best')
    plt.xlabel("Conditional transmission probability μ")
    plt.ylabel("transport probability")
    plt.show()
