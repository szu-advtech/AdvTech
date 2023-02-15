import numpy as np
from Function.utilities.Reject import *
from Function.utilities import *

def ExML_train(y_tr,X_tr,X_tr_all,model_org,theta,C_h,C_g,candidate_set,budget,hidden_class_id):
    # parameters setting：

    opts = {}
    opts['kernel_type'] = 'Gauss'
    opts['manifold'] = 'off'
    K = max(np.shape(candidate_set))
    X_pool = [0]*K
    y_pool = [0]*K
    T = np.ceil(np.log(K) / np.log(2))
    index_pool = [0]*T
    model_pool = [0]*K
    sample_pool = np.zeros((K, 1))
    remain_index = np.array(list(range(np.shape(X_tr)[0])))
    active_set = np.array(list(range(K)))
    loss_set = np.ones((K, 1)) * -1

    for t in range(T):
        # sampling $n_t$ samples from the training dataset：
        n_t = np.floor(budget / T / max(np.shape(active_set)))
        try:
            index_selected = np.randsample(remain_index, min(n_t, max(np.shape(remain_index))))
        except:
            break

        remain_index = remain_index[not np.ismember(remain_index, index_selected)]

        # updating the active candidates set
        for feature_index in active_set:
            # querying active features
            X_i = np.append(X_tr[index_selected][:], X_tr_all[index_selected][candidate_set[feature_index]], axis=0)
            y_i = y_tr[index_selected]

            # updating $D_i$ with selected samples
            X_pool[feature_index] = np.append(X_pool[feature_index],X_i,axis=1)
            y_pool[feature_index] = np.append(y_pool[feature_index],y_i,axis=1)
            sample_pool[feature_index] = sample_pool[feature_index] + n_t

            # retraining a model on $D_i$
            M = pdist2(X_pool[feature_index], X_pool[feature_index])
            opts.kernel_para = np.median(M[:])
            [model_aug, labelSet] = rejectsvmtrain(y_pool[feature_index], X_pool[feature_index], theta, C_h, C_g, opts)
            model_pool[feature_index] = model_aug

            # calculate the empirical score $\hat{R}_{D_i}:
            loss_set[feature_index] = indicator_loss(y_pool[feature_index], X_pool[feature_index], model_aug, theta, labelSet, hidden_class_id)

        # update active featue_set
        loss_active = loss_set(active_set)
        [a, index] = np.sort(loss_active)
        remain_num = np.ceil(max(np.shape(active_set)) / 2)
        active_set = active_set[index[:remain_num]]

    selected_feature_index = active_set
    train_num = np.shape(X_pool[selected_feature_index])[0]
    model_aug = model_pool[selected_feature_index]
    cascade_model = [model_org, model_aug]

    return [cascade_model,labelSet,selected_feature_index,train_num,sample_pool]
