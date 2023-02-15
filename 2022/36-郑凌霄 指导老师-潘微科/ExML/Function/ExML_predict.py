import numpy as np
from Function.utilities.Reject import *

def ExML_predict(y_te,X_te,cascade_model,labelSet,hidden_class_id,org_feature):
# The first layer of model cascade: predicting with the oringal model

    model_org = cascade_model[0]
    M = pdist2(X_te[:][0: max(np.shape(org_feature))],model_org['X'])
    if(model_org['opts']['kernel_type'] == 'Gauss'):
        K = Gauss_kernel(M,model_org['opts']['kernel_para'])
    else:
        print('error')
    pred_h = np.dot(K, model_org['w'])
    pred_r = np.dot(K, model_org['u'])
    prediction_org = np.zeros(np.shape(pred_h))
    index_low_conf = (pred_r < 0)
    index_1 = (pred_r >= 0) & (pred_h > 0)
    index_2 = (pred_r >= 0) & (pred_h <= 0)
    prediction_org[index_low_conf] = hidden_class_id
    prediction_org[index_1] = labelSet[0]
    prediction_org[index_2] = labelSet[1]

    # The second layer of model cascade: refine the prediction with
    # augmetned mode
    model_aug = cascade_model[1]
    M = pdist2(X_te, model_aug.X)
    if (model_aug['opts']['kernel_type'] == 'Gauss'):
        K = Gauss_kernel(M, model_aug['opts']['kernel_para'])
    else:
        print('error')

    pred_h = np.dot(K, model_aug['w'])
    pred_r = np.dot(K, model_aug['u'])
    prediction_aug = np.zeros(np.shape(pred_h))

    index_new = (pred_r < 0)
    index_1 = (pred_r >= 0) & (pred_h > 0)
    index_2 = (pred_r >= 0) & (pred_h <= 0)
    prediction_aug[index_new] = hidden_class_id
    prediction_aug[index_1] = labelSet[0]
    prediction_aug[index_2] = labelSet[1]

    predict_cascade = prediction_org
    predict_cascade[index_low_conf] = prediction_aug[index_low_conf]

    return [predict_cascade,prediction_aug,prediction_org]
