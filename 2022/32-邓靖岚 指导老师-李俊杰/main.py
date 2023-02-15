from __future__ import print_function
from __future__ import unicode_literals
from builtins import int

import datetime
import errno
import h5py
import itertools
import json
import numpy as np
import os
import sys
import time
import h5py
import cProfile

from mpi4py import MPI

# MPI 多进程通信
comm = MPI.COMM_WORLD
# 主进程
rank = comm.rank
from scipy import linalg

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pylab

from truncated_GMM import TruncatedGaussianMixture as GMM
from utils.data import get_data
import utils.coreset as cs

# changed place :
# The type of C_per_dim in data.py : From C_per_dim = np.round(C ** (1. / D)).astype(np.int32) to np.int64
# Get h5file in data.py: From h5file['train/data'].value to h5file['train/data']


# ===== Set Model Parameters =====================
# default parameters
params = {
    'algorithm': 'var-GMM-S',
    'C': 400,
    'Cprime': 5,
    'G': 5,
    'Niter': 25,
    'Ninit': 5,
    'dataset': 'BIRCH2-400',
    'VERBOSE': True,
}


# parameters testing
user_params = {'algorithm' : 'var-GMM-S+1'}

# parameters given by user via command line
try:
    user_params = {'algorithm': sys.argv[1]}
    if len(sys.argv) > 2:
        user_params.update(dict(arg.split('=', 1) for arg in sys.argv[2:]))
except:
    print('WARNING: Could not read user parameters properly. Reverting to default parameters.')
    user_params = {}

params.update(user_params)
params['C'] = int(params['C'])
params['Cprime'] = int(params['Cprime'])
params['G'] = int(params['G'])
params['Niter'] = int(params['Niter'])
params['Ninit'] = int(params['Ninit'])
params['VERBOSE'] = True if (params['VERBOSE'] == True or params['VERBOSE'] == 'True') else False

# define the outputs
if params['VERBOSE']:
    params['VERBOSE'] = {
        'll': True,  # loglikelihood
        'fe': True,  # free energy
        'qe': True,  # quantization error
        'cs': True,  # clustering scores (purity, NMI, AMI)
        'nd': True,  # number of distance evaluations
        'np': 1,  # picture output every n iterations
    }
else:
    params['VERBOSE'] = {
        'll': False,
        'fe': False,
        'qe': False,
        'cs': False,
        'nd': True,
        'np': np.inf,
    }

# ===== Instantiate Model ========================
gmm = GMM(params)

# ===== Load Data ================================
# X: data
# Y: labels
# data_params: parameters of dataset and so on
# gt_values: [means, sigma, covariances]
X, Y, data_params, gt_values = get_data(params['dataset'], comm)
print("X from h5:", X)
print("Y from h5:", Y)

# ===== Construction of Coreset ==================
# coreset class : reading path to dataset
c = cs.Coreset(params['dataset'])

# load data
# X is dataset, and can be read by X[:,:]
c.load_data(X, Y)

# construct coreset by parameters
# c.construct(cluster = params['C'], delta = 0.05, epsilon = 0.4)

# construct coreset by user definition
# c.construct(50000)

# get coreset data and label(if exists)
Xc = c.coreset
Yc = c.coreset_label
# c.construct(80000)
# print("Shape of coreset", c.coreset.shape)
# c_size = c.coreset.shape[0]
# print(c.coreset[:,:])
# (Xc,Yc) = c.writeH5()


# ===== Initialize Output ========================
# calculate ground-truth scores
VERBOSE = params['VERBOSE']


# Make sure that VERBOSE['attr'] and gt_values['means' is not none
# if VERBOSE['ll'] and gt_values['means'] is not None and gt_values['sigma_sq'] is not None
#     loglikelihood_gt = gmm.loglikelihood(
#         X,
#         gt_values
#     )
# else loglikelihood_gt = 'not available'

# 求似然函数Compute log-likelihood function：\sum {\gamma_n * \log p(y_n|\Theta)}
loglikelihood_gt = gmm.loglikelihood(
    Xc,
    gt_values
) if VERBOSE['ll'] and gt_values['means'] is not None and gt_values['sigma_sq'] is not None else 'not available'
# 求泛化误差：\sum ||X-E[X]||
qe_gt = gmm.quantization_error(
    Xc,
    gt_values['means'],
) if VERBOSE['qe'] and gt_values['means'] is not None else 'not available'
# 聚类指标
purity_score_gt, NMI_score_gt, AMI_score_gt = gmm.clustering_scores(
    Xc,
    Yc,
    gt_values['means'],
) if VERBOSE['cs'] and gt_values['means'] is not None else ('not available', 'not available', 'not available')

if rank == 0:
    print('data set: ', params['dataset'])
    print('#samples: ', Xc.shape[0])
    print('#features: ', Xc.shape[1])
    print('algorithm: ', params['algorithm'])
    print('C: ', params['C'])
    print('Cprime: ', params['Cprime'])
    print('G: ', params['G'])
    print('#Iterations: ', params['Niter'])
    print('#Initial E-Step Iterations: ', params['Ninit'])

    # create files
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    while True:
        filename = './output/{}/{}/C{}_K{}_G{}_N{}-{}/{}'.format(
            data_params['dataset_name'],
            params['algorithm'],
            str(params['C']), str(params['Cprime']), str(params['G']),
            str(params['Niter']), str(params['Ninit']),
            datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        directory = os.path.dirname(filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
        try:
            file_handle = os.open(filename + '_results.txt', flags)
        except OSError as e:
            if e.errno == errno.EEXIST:  # Failed as the file already exists.
                pass
            else:  # Something unexpected went wrong so reraise the exception.
                raise
        else:  # No exception, so the file must have been created successfully.
            if VERBOSE['ll']: print('ground-truth LogLikelihood: ', loglikelihood_gt)
            if VERBOSE['qe']: print('ground truth Q-error:', qe_gt)
            if VERBOSE['cs']: print('ground truth Purity:', purity_score_gt)
            if VERBOSE['cs']: print('ground truth NMI:', NMI_score_gt)
            if VERBOSE['cs']: print('ground truth AMI:', AMI_score_gt)
            with open(filename + '_results.txt', 'w') as file:
                if VERBOSE['ll']: file.write('#ground truth LogLikelihood: ' + str(loglikelihood_gt) + '\n')
                if VERBOSE['qe']: file.write('#ground truth Q-error: ' + str(qe_gt) + '\n')
                if VERBOSE['cs']: file.write('#ground truth Purity: ' + str(purity_score_gt) + '\n')
                if VERBOSE['cs']: file.write('#ground truth NMI: ' + str(NMI_score_gt) + '\n')
                if VERBOSE['cs']: file.write('#ground truth AMI: ' + str(AMI_score_gt) + '\n')
                outstr = ('{:' + str(
                    int(np.log10(params['Niter']) + 1)) + '}\t{:15}\t{:15}\t{:15}\t{:8}\t{:8}\t{:8}\t{:8}\n').format(
                    'n', 'Free Energy', 'LogLikelihood',
                    'Q-Error', 'Purity', 'NMI', 'AMI', '#D-Evals (Speed-Up)'
                )
                file.write(outstr)


            def default(o):
                if isinstance(o, np.int64): return int(o)
                raise TypeError


            json.dump([params, data_params], open(filename + "_parameters.txt", 'w'), default=default)
            break
else:
    filename = None

# ===== Fit Model ================================
# fit model using full dataset
# gmm.fit(X, Y, filename=filename, origin_X=X, plot=True)
# json.dump({"coreset" : False}, open(filename + "_parameters.txt", 'a'))

# fit model using coreset
gmm.fit(Xc, Yc, filename=filename, origin_X=X, plot=True)

# ===== Output Results ===========================
if rank == 0:
    h5f = h5py.File(''.join((filename, '.hdf5')), 'w')
    h5f.create_dataset('means', data=gmm.means)
    h5f.create_dataset('sigma_sq', data=gmm.sigma_sq)
    h5f.close()
