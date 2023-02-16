import numpy as np

def Gauss_kernel(M,para):

    return np.exp(-M**2/2/para**2)
