# -*- coding: utf-8 -*-
"""
@author: H-divergence: A Decision-Theoretic Discrepancy Measure for Two Sample Tests
@Implementation of H-Divergence two sample test in our paper on HDGM dataset

BEFORE USING THIS CODE:
Numpy and Sklearn are required. Users can install Python via Anaconda to obtain both packages.
Anaconda can be found in https://www.anaconda.com/distribution/#download-section.

This code is based on the two sample test experiment in "Learning Deep Kernels for Non-Parametric Two-Sample Tests".
The original github repo is https://github.com/fengliu90/DK-for-TST, please also follow their instruction to reproduce the baselines in our paper.
"""
import numpy as np
import torch
import argparse
parser = argparse.ArgumentParser()
from utils import JSV_Gaussian
import os

# parameters to generate data
parser.add_argument('--n', type=int, default=1000, help="number of samples per mode")
parser.add_argument('--d', type=int, default=10, help="dimension of samples (default: 10)")
parser.add_argument('--ntrial', type=int, default=10, help="number of trials (default: 10)")
parser.add_argument('--exptype', type=str, default="power", help="type of experiment (power or typei) (default: power)")
parser.add_argument('--vtype', type=str, default="vjs", help="type of experiment (vjs or vmin) (default: vjs)")
parser.add_argument('--output', type=str, default=".", help="output directory (default: current directory)")
args = parser.parse_args()

# Setup seeds
np.random.seed(1102)
# Setup for experiments
dtype = np.float
N_per = 100 # permutation times
alpha = 0.05 # test threshold
d = args.d # dimension of data
n = args.n # number of samples in per mode
print('n: '+str(n)+' d: '+str(d))
K = args.ntrial # number of trials
N = 100 # # number of test sets
N_f = 100.0 # number of test sets (float)

# Generate variance and co-variance matrix of Q
Num_clusters = 2 # number of modes
n_components = Num_clusters*2
mu_mx = np.zeros([Num_clusters,d])
mu_mx[1] = mu_mx[1] + 0.5
sigma_mx_1 = np.identity(d)
sigma_mx_2 = [np.identity(d),np.identity(d)]
sigma_mx_2[0][0,1] = 0.5
sigma_mx_2[0][1,0] = 0.5
sigma_mx_2[1][0,1] = -0.5
sigma_mx_2[1][1,0] = -0.5
s1 = np.zeros([n*Num_clusters, d])
s2 = np.zeros([n*Num_clusters, d])

# Naming variables
Results = np.zeros([1,K])

# Repeat experiments K times (K = 10) and report average test power (rejection rate)
for kk in range(K):
    # Generate HDGM-D
    for i in range(Num_clusters):
        np.random.seed(seed=1102*kk + i + n)
        s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
    for i in range(Num_clusters):
        np.random.seed(seed=819*kk + 1 + i + n)
        if args.exptype == "power":
            s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
        elif args.exptype == "typei":
            s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
            # for validating type-I error (s1 ans s2 are from the same distribution)
        else:
            raise NotImplementedError("Please choose either power or typei experiment")

    # Compute test power
    H_u = np.zeros(N)
    T_u = np.zeros(N)
    M_u = np.zeros(N)
    np.random.seed(1102)
    count_u = 0
    for k in range(N):
        # Generate HDGM
        for i in range(Num_clusters):
            np.random.seed(seed=1102 * (k+2) + 2*kk + i + n)
            s1[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
        for i in range(Num_clusters):
            np.random.seed(seed=819 * (k + 1) + 2*kk + i + n)
            if args.exptype == "power":
                s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_2[i], n)
            elif args.exptype == "typei":
                s2[n * (i):n * (i + 1), :] = np.random.multivariate_normal(mu_mx[i], sigma_mx_1, n)
            # for validating type-I error (s1 ans s2 are from the same distribution)
            else:
                raise NotImplementedError("Please choose either power or typei experiment")
        S = np.concatenate((s1, s2), axis=0)
        # Run two sample test on generated data
        h_u, threshold_u, jsv_u = JSV_Gaussian(S, N_per, n*Num_clusters, alpha, n_components, dtype, args.vtype)
        # Gather results
        count_u = count_u + h_u
        if k % 10 == 0:
            print(kk, k, "V-Div GMM:", count_u)
        H_u[k] = h_u
        T_u[k] = threshold_u
        M_u[k] = jsv_u
    # Print test power of MMD-D
    print("Test Power of V-Div GMM on HDGM: ", H_u.sum() / N_f)
    Results[0, kk] = H_u.sum() / N_f
    print("Test Power of V-Div GMM on HDGM (K times): ", Results[0])
    print("Average Test Power of V-Div GMM on HDGM: ", Results[0].sum() / (kk + 1))
np.save(os.path.join(args.output, f"HDGM_{args.exptype}_n{n}_d{d}_ncomp{n_components}_{args.vtype}_GMM"),Results)
