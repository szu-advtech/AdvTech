import math
import networkx as nx
from scipy import stats
import random
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import pandas as pd

import InitG as ig
import KStest as ks
import KLtest as kl
import DrawGraph as dg
import RW
import SamplingMethods as sm
import Util as util
import SGP

matplotlib.use('TkAgg')  # 高版本py必须加上，否则报错


def sampling(G=nx.Graph()):
    G_degree_distribution = ks.degree_distribution(G)
    G_hop_plot_distribution = ks.hop_plot_distribution(G)
    G_cc_distribution = ks.clustering_coefficient_distribution(G)
    G_k_core_distribution = ks.k_core_distribution(G)

    GS1_degree_ks = 0
    GS2_degree_ks = 0
    GS3_degree_ks = 0
    GS4_degree_ks = 0
    GS5_degree_ks = 0
    GS6_degree_ks = 0

    GS1_hop_plot_ks = 0
    GS2_hop_plot_ks = 0
    GS3_hop_plot_ks = 0
    GS4_hop_plot_ks = 0
    GS5_hop_plot_ks = 0
    GS6_hop_plot_ks = 0

    GS1_clustering_coefficient_ks = 0
    GS2_clustering_coefficient_ks = 0
    GS3_clustering_coefficient_ks = 0
    GS4_clustering_coefficient_ks = 0
    GS5_clustering_coefficient_ks = 0
    GS6_clustering_coefficient_ks = 0

    GS1_k_core_ks = 0
    GS2_k_core_ks = 0
    GS3_k_core_ks = 0
    GS4_k_core_ks = 0
    GS5_k_core_ks = 0
    GS6_k_core_ks = 0

    GS1_degree_kl = 0
    GS2_degree_kl = 0
    GS3_degree_kl = 0
    GS4_degree_kl = 0
    GS5_degree_kl = 0
    GS6_degree_kl = 0

    GS1_hop_plot_kl = 0
    GS2_hop_plot_kl = 0
    GS3_hop_plot_kl = 0
    GS4_hop_plot_kl = 0
    GS5_hop_plot_kl = 0
    GS6_hop_plot_kl = 0

    GS1_clustering_coefficient_kl = 0
    GS2_clustering_coefficient_kl = 0
    GS3_clustering_coefficient_kl = 0
    GS4_clustering_coefficient_kl = 0
    GS5_clustering_coefficient_kl = 0
    GS6_clustering_coefficient_kl = 0

    GS1_k_core_kl = 0
    GS2_k_core_kl = 0
    GS3_k_core_kl = 0
    GS4_k_core_kl = 0
    GS5_k_core_kl = 0
    GS6_k_core_kl = 0

    for i in range(0, 3):
        GS1 = RW.Meropolis_Hastings_RW(G.copy())
        GS2 = RW.Meropolis_Hastings_RJ(G.copy())
        GS3 = RW.RW(G.copy())
        GS4 = RW.RJ(G.copy())
        GS5 = sm.FFS(G.copy())
        GS6 = SGP.SGP(G.copy())

        GS1_degree_distribution = ks.degree_distribution(GS1)
        GS2_degree_distribution = ks.degree_distribution(GS2)
        GS3_degree_distribution = ks.degree_distribution(GS3)
        GS4_degree_distribution = ks.degree_distribution(GS4)
        GS5_degree_distribution = ks.degree_distribution(GS5)
        GS6_degree_distribution = ks.degree_distribution(GS6)

        GS1_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS1_degree_distribution)
        GS2_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS2_degree_distribution)
        GS3_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS3_degree_distribution)
        GS4_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS4_degree_distribution)
        GS5_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS5_degree_distribution)
        GS6_degree_ks += ks.KolmogorowSmirnov(G_degree_distribution, GS6_degree_distribution)

        GS1_hop_plot_distribution = ks.hop_plot_distribution(GS1)
        GS2_hop_plot_distribution = ks.hop_plot_distribution(GS2)
        GS3_hop_plot_distribution = ks.hop_plot_distribution(GS3)
        GS4_hop_plot_distribution = ks.hop_plot_distribution(GS4)
        GS5_hop_plot_distribution = ks.hop_plot_distribution(GS5)
        GS6_hop_plot_distribution = ks.hop_plot_distribution(GS6)

        GS1_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS1_hop_plot_distribution)
        GS2_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS2_hop_plot_distribution)
        GS3_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS3_hop_plot_distribution)
        GS4_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS4_hop_plot_distribution)
        GS5_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS5_hop_plot_distribution)
        GS6_hop_plot_ks += ks.hop_plot_KS(G_hop_plot_distribution, GS6_hop_plot_distribution)

        GS1_cc_distribution = ks.clustering_coefficient_distribution(GS1)
        GS2_cc_distribution = ks.clustering_coefficient_distribution(GS2)
        GS3_cc_distribution = ks.clustering_coefficient_distribution(GS3)
        GS4_cc_distribution = ks.clustering_coefficient_distribution(GS4)
        GS5_cc_distribution = ks.clustering_coefficient_distribution(GS5)
        GS6_cc_distribution = ks.clustering_coefficient_distribution(GS6)

        GS1_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS1_cc_distribution)
        GS2_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS2_cc_distribution)
        GS3_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS3_cc_distribution)
        GS4_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS4_cc_distribution)
        GS5_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS5_cc_distribution)
        GS6_clustering_coefficient_ks += ks.clustering_coefficient_KS(G_cc_distribution, GS6_cc_distribution)

        GS1_k_core_distribution = ks.k_core_distribution(GS1)
        GS2_k_core_distribution = ks.k_core_distribution(GS2)
        GS3_k_core_distribution = ks.k_core_distribution(GS3)
        GS4_k_core_distribution = ks.k_core_distribution(GS4)
        GS5_k_core_distribution = ks.k_core_distribution(GS5)
        GS6_k_core_distribution = ks.k_core_distribution(GS6)

        GS1_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS1_k_core_distribution)
        GS2_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS2_k_core_distribution)
        GS3_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS3_k_core_distribution)
        GS4_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS4_k_core_distribution)
        GS5_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS5_k_core_distribution)
        GS6_k_core_ks += ks.k_core_KS(G_k_core_distribution, GS6_k_core_distribution)

        GS1_degree_kl += kl.degree_KL(G, GS1)
        GS2_degree_kl += kl.degree_KL(G, GS2)
        GS3_degree_kl += kl.degree_KL(G, GS3)
        GS4_degree_kl += kl.degree_KL(G, GS4)
        GS5_degree_kl += kl.degree_KL(G, GS5)
        GS6_degree_kl += kl.degree_KL(G, GS6)

        GS1_hop_plot_kl += kl.hop_plot_KL(G_hop_plot_distribution, GS1_hop_plot_distribution)
        GS2_hop_plot_kl += kl.hop_plot_KL(G_hop_plot_distribution, GS2_hop_plot_distribution)
        GS3_hop_plot_kl += kl.hop_plot_KL(G_hop_plot_distribution, GS3_hop_plot_distribution)
        GS4_hop_plot_kl += kl.hop_plot_KL(G_hop_plot_distribution, GS4_hop_plot_distribution)
        GS5_hop_plot_kl += kl.hop_plot_KL(G_hop_plot_distribution, GS5_hop_plot_distribution)
        GS6_hop_plot_kl += kl.hop_plot_KL(G_hop_plot_distribution, GS6_hop_plot_distribution)

        GS1_clustering_coefficient_kl += kl.clustering_coefficient_KL(G_cc_distribution, GS1_cc_distribution)
        GS2_clustering_coefficient_kl += kl.clustering_coefficient_KL(G_cc_distribution, GS2_cc_distribution)
        GS3_clustering_coefficient_kl += kl.clustering_coefficient_KL(G_cc_distribution, GS3_cc_distribution)
        GS4_clustering_coefficient_kl += kl.clustering_coefficient_KL(G_cc_distribution, GS4_cc_distribution)
        GS5_clustering_coefficient_kl += kl.clustering_coefficient_KL(G_cc_distribution, GS5_cc_distribution)
        GS6_clustering_coefficient_kl += kl.clustering_coefficient_KL(G_cc_distribution, GS6_cc_distribution)

        GS1_k_core_kl += kl.k_core_KL(G_k_core_distribution, GS1_k_core_distribution)
        GS2_k_core_kl += kl.k_core_KL(G_k_core_distribution, GS2_k_core_distribution)
        GS3_k_core_kl += kl.k_core_KL(G_k_core_distribution, GS3_k_core_distribution)
        GS4_k_core_kl += kl.k_core_KL(G_k_core_distribution, GS4_k_core_distribution)
        GS5_k_core_kl += kl.k_core_KL(G_k_core_distribution, GS5_k_core_distribution)
        GS6_k_core_kl += kl.k_core_KL(G_k_core_distribution, GS6_k_core_distribution)

    print('degree_KS------------------------')
    print(GS1_degree_ks / 3)
    print(GS2_degree_ks / 3)
    print(GS3_degree_ks / 3)
    print(GS4_degree_ks / 3)
    print(GS5_degree_ks / 3)
    print(GS6_degree_ks / 3)
    print()
    print('hop_plot_KS----------------------')
    print(GS1_hop_plot_ks / 3)
    print(GS2_hop_plot_ks / 3)
    print(GS3_hop_plot_ks / 3)
    print(GS4_hop_plot_ks / 3)
    print(GS5_hop_plot_ks / 3)
    print(GS6_hop_plot_ks / 3)
    print()
    print('clustering_coefficient_KS----------')
    print(GS1_clustering_coefficient_ks / 3)
    print(GS2_clustering_coefficient_ks / 3)
    print(GS3_clustering_coefficient_ks / 3)
    print(GS4_clustering_coefficient_ks / 3)
    print(GS5_clustering_coefficient_ks / 3)
    print(GS6_clustering_coefficient_ks / 3)
    print()
    print('k_core_KS----------------------')
    print(GS1_k_core_ks / 3)
    print(GS2_k_core_ks / 3)
    print(GS3_k_core_ks / 3)
    print(GS4_k_core_ks / 3)
    print(GS5_k_core_ks / 3)
    print(GS6_k_core_ks / 3)
    print()
    print('degree_KL-----------------------------')
    print(GS1_degree_kl / 3)
    print(GS2_degree_kl / 3)
    print(GS3_degree_kl / 3)
    print(GS4_degree_kl / 3)
    print(GS5_degree_kl / 3)
    print(GS6_degree_kl / 3)
    print()
    print('hop_plot_KL-----------------------------')
    print(GS1_hop_plot_kl / 3)
    print(GS2_hop_plot_kl / 3)
    print(GS3_hop_plot_kl / 3)
    print(GS4_hop_plot_kl / 3)
    print(GS5_hop_plot_kl / 3)
    print(GS6_hop_plot_kl / 3)
    print()
    print('clustering_coefficient_KL-----------------------------')
    print(GS1_clustering_coefficient_kl / 3)
    print(GS2_clustering_coefficient_kl / 3)
    print(GS3_clustering_coefficient_kl / 3)
    print(GS4_clustering_coefficient_kl / 3)
    print(GS5_clustering_coefficient_kl / 3)
    print(GS6_clustering_coefficient_kl / 3)
    print()
    print('k_core_KL-----------------------------')
    print(GS1_k_core_kl / 3)
    print(GS2_k_core_kl / 3)
    print(GS3_k_core_kl / 3)
    print(GS4_k_core_kl / 3)
    print(GS5_k_core_kl / 3)
    print(GS6_k_core_kl / 3)
    print()
    # dg.drawCDF(G, GS1, GS2, GS3)


def CDFtest(G=nx.Graph()):
    GS1 = RW.Meropolis_Hastings_RW(G.copy())
    GS2 = RW.Meropolis_Hastings_RJ(G.copy())
    GS3 = RW.RW(G.copy())
    GS4 = RW.RJ(G.copy())
    GS5 = sm.FFS(G.copy())
    GS6 = SGP.SGP(G.copy())
    dg.drawCDF(G, GS1, GS2, GS3, GS4, GS5, GS6)


G = ig.email_univ()
sampling(G)