from scipy.spatial.distance import pdist, squareform
from .Gauss_kernel import Gauss_kernel

def PSD(X,opts):
# This function is used for calculating the PSD kernel

    if (opts['kernel_type'] == 'Gauss'):
        M = squareform(pdist(X))
        K = Gauss_kernel(M,opts['kernel_para'])
        return K
    else:
        print('error')
