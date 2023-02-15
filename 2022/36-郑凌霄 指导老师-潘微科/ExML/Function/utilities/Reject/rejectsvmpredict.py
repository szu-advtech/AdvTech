import numpy as np
from .Gauss_kernel import Gauss_kernel

def pdist2(A, B):

    H = np.shape(A)[0]
    W = np.shape(B)[0]
    C = np.zeros((H,W))

    for i in range(H):
        for j in range(W):
            C[i][j] = np.sqrt(np.sum((A[i] - B[j]) ** 2))

    return C

def rejectsvmpredict(y_te,X_te,model,labelSet,new_class_id):
#This function is used for predicting with the rejection model

    M = pdist2(X_te,model['X'])
    if(model['opts']['kernel_type'] == 'Gauss'):
        K = Gauss_kernel(M,model['opts']['kernel_para'])
    else:
        print('error')

    pred_h = np.dot(K,model['w'])
    pred_r = np.dot(K,model['u'])
    prediction = np.zeros((np.shape(pred_h)[0],1))
    index_new = pred_r < 0
    index_1 = (pred_r>=0) & (pred_h>0)
    index_2 = (pred_r>=0) & (pred_h<=0)
    prediction[index_new] = new_class_id
    prediction[index_1] = labelSet[0]
    prediction[index_2] = labelSet[1]

    return [prediction, pred_h, pred_r]

