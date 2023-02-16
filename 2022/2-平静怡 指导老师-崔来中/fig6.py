import numpy as np
from matplotlib import pyplot as plt
import analy_model
import sim_csma_tp2

if __name__ == "__main__":
    miu1,csma_tp1=analy_model.csma_tp(0.5)
    miu2, csma_tp2 = analy_model.csma_tp(0.05)
    plt.plot(miu1, csma_tp1, 'b--', label='Analytical Model for λ=0.5')
    plt.plot(miu1, csma_tp2, 'r--', label='Analytical Model for λ=0.05')
    miu3,sim_csma_tp1=sim_csma_tp2.sim_csma_tp(0.5)
    miu4, sim_csma_tp2 = sim_csma_tp2.sim_csma_tp(0.05)
    plt.plot(miu3, sim_csma_tp1, 'darkorange', label='Simulation for λ=0.5')
    plt.plot(miu4, sim_csma_tp2, label='Simulation for λ=0.05')
    plt.legend(loc='best')
    plt.xlabel("Conditional transmission probability μ")
    plt.ylabel("transport probability")
    plt.show()