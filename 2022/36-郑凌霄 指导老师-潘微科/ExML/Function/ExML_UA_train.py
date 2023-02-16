import numpy as np
from Function.utilities.Reject import *
from Function.utilities import *

def ExML_UA_train(y_tr,X_tr,X_tr_all,model_org,theta,C_h,C_g,candidate_set,budget,hidden_class_id):
    # parameters setting：
    opts={}
    opts['kernel_type'] = 'Gauss'
    opts['manifold'] = 'off'
    K = max(np.shape(candidate_set))
    model_pool = [0]*K
    remain_index = [0]*np.shape(X_tr)[0]
    loss_set = np.ones((K, 1)) * -1

    # uniformly allocate the budget
    index_selected = np.randsample(remain_index, np.floor(budget / K))
    train_num = max(np.shape(index_selected))
    for feature_index in range(K):
        # updating $D_i$ with selected samples
        X_i = np.append(X_tr[index_selected][:], X_tr_all[index_selected][candidate_set[feature_index]],axis=0)
        y_i = y_tr[index_selected]

        # retraining a model on $D_i$
        M = pdist2(X_i, X_i)
        opts['kernel_para'] = np.median(M[:])
        [model_aug, labelSet] = rejectsvmtrain(y_i, X_i, theta, C_h, C_g, opts)
        model_pool[feature_index] = model_aug

        # calculate the empirical score $\hat{R}_{D_i}:
        loss_set[feature_index] = indicator_loss(y_i, X_i, model_aug, theta, labelSet, hidden_class_id)

    [a, index] = np.sort(loss_set)
    selected_feature_index = index(0)
    model_aug = model_pool[selected_feature_index]
    cascade_model = [model_org, model_aug]

    return [cascade_model,labelSet,selected_feature_index,train_num,loss_set]
