import numpy as np
from Function.utilities.Reject import *

def indicator_loss(y,X,model,theta,labelSet,hidden_class_id):
# This is the function used for calculate the empirical score to measure
# the quailty of the candidate features

    M = pdist2(X, model['X'])
    if(model['opts']['kernel_type'] == 'Gauss'):
        K = Gauss_kernel(M, model['opts']['kernel_para'])
    else:
        print('error')

    pred_h = np.dot(K, model['w'])
    pred_r = np.dot(K, model['u'])
    # calculate the score
    prediction = np.zeros((np.shape(pred_h)[0],1))
    index_new = pred_r < 0
    index_high = pred_r > 0
    index_1 = (pred_r>=0) & (pred_h>0)
    index_2 = (pred_r>=0) & (pred_h<=0)
    prediction[index_new] = hidden_class_id
    prediction[index_1] = labelSet[0]
    try:
        prediction[index_2] = labelSet[1]
    except:
        print('error')

    loss = theta * sum(index_new) + sum(prediction[index_high] == y[index_high])
    loss = loss / np.shape(index_new)[0]

    return loss
