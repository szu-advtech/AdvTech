import numpy as np
import scipy.io
import os
from Function.utilities.Reject import *
from Function.utilities import *
from Function import *

def main():
    # add various dataset configurations including
    # - different orginal feature indicated by 'org'
    # - different known and unknowns class partition indicated by 'config'

    data_set_org = ['Mfeat']
    data_set = []
    for config in range(5):
        for i in [2]:
            data_set.append = str(data_set_org)+ '_org'+ str(i)+ 'config_'+ str(config)


    # contenders
    alg_set = ["org", "augUA", "augME", "UA", "ME"]

    # parameters setting
    acc_min = 0.95
    theta_set = [0.1, 0.2, 0.3, 0.4]
    repeat = 10
    budget_rio_set = [0.1, 0.2, 0.3]
    hidden_class_id = 10000
    acc_recorder = np.zeros(max(np.shape(alg_set)) + 1, 10, max(np.shape(theta_set)))

    # running the contenders with various dataset configuration
    for data in data_set:
        for budget_rio in budget_rio_set:
            data = str(data)
            Dataset = scipy.io.loadmat('./Data/Dataset/' + data_set_org + '/'+ data)
            Index = scipy.io.loadmat('./Data/Index/' + data_set_org + '/'+ data)

            candidate_set = Dataset['candidate_set']
            org_feature = Dataset['org_feature']
            X = Dataset['X']
            y_obs = Dataset['y_obs']
            y_true = Dataset['y_true']

            train_index = Index['train_index']
            test_index = Index['test_index']

            # initilize recorders for results
            for alg in alg_set:
                eval('acc_' + str(alg) + '= np.zeros((np.shape(train_index)[0],1))')

            best_recorder = np.zeros(max(np.shape(candidate_set)), 10, max(np.shape(theta_set)))
            ME_feature_pool = np.zeros((np.shape(train_index)[0], 1))
            UA_feature_pool = np.zeros((np.shape(train_index)[0], 1))
            budget = np.floor(max(np.shape(candidate_set)) * budget_rio * np.shape(train_index)[1])
            print('Mfeat:'+ data+ ' with budget ratio '+ str(budget_rio)+ ' and the orignial feature Fou')

            # running 10 sample partitions for each configuration
            for iteration in range(np.size(np.shape(train_index)[0])):
                print('|--sample partition:' + str(iteration+1))
                # obtain the training data
                X_tr = X[train_index[iteration][:]][org_feature]
                X_tr_all = X[train_index[iteration][:]][:]
                y_tr = y_obs[train_index[iteration][:]]
                y_true_tr = y_true[train_index[iteration][:]]

                # obtain the testing data
                X_te = X[test_index[iteration][:]][org_feature]
                X_te_all = X[test_index[iteration][:]][:]
                y_te = y_obs[test_index[iteration][:]]
                y_true_te = y_true[test_index[iteration][:]]


                # training the first layer of ExML to ensure 95% accuracy on known classes
                print('|----training for the first layer of ExML')
                opts = {}
                opts['kernel_type'] = 'Gauss'
                [theta_org] = SVM_reject_valid(y_tr, X_tr, [0.55, 0.95], acc_min, hidden_class_id, opts, 3)
                opts.kernel_type = 'Gauss'
                M = pdist2(X_tr, X_tr)
                opts.kernel_para = np.median(M[:])
                [model_search, a] = rejectsvmtrain(y_tr, X_tr, theta_org, 1, 1, opts)

                # training SL and variants of ExML with various theta, where the best on is selected
                theta_index = 1
                print('|----training SL and variants of ExML with various theta:')
                for theta in theta_set:
                    print(str(theta) + ',',end='')
                    # train with SL
                    opts['ernel_type'] = 'Gauss'
                    opts['manifold'] = 'off'
                    M = pdist2(X_tr, X_tr)
                    opts['kernel_para'] = np.median(M[:])
                    [model_reject, labelSet] = rejectsvmtrain(y_tr, X_tr, theta, 1, 1, opts)
                    [predict_org,a,a] = rejectsvmpredict(y_te, X_te, model_reject, labelSet, hidden_class_id)
                    # print('SL')

                    # train with ExML using uniform allocation (UA)
                    [UA_model, labelSet, UA_feature_index, train_UA, loss_set] = ExML_UA_train(y_tr, X_tr, X_tr_all, model_search, theta, 1, 1, candidate_set, budget, hidden_class_id)
                    X_te_aug = X[test_index[iteration][:]][np.append(org_feature, candidate_set[UA_feature_index],axis=0)]
                    [predict_UA, predict_augUA, predict_search] = ExML_predict(y_te, X_te_aug, UA_model, labelSet, hidden_class_id,org_feature)
                    # print('UA')

                    # train with ExML using median elimination (ME)
                    [ME_model, labelSet, ME_feature_index, train_ME, sample_pool] = ExML_train(y_tr, X_tr, X_tr_all, model_search, theta, 1, 1, candidate_set, budget, hidden_class_id)
                    X_te_aug = X[test_index[iteration][:], np.append(org_feature, candidate_set[ME_feature_index],axis=0)]
                    [predict_ME, predict_augME] = ExML_predict(y_te, X_te_aug, ME_model, labelSet, hidden_class_id, org_feature)
                    # print('ME')

                    # record the results including accuracy and index of selected candidate features
                    for alg in alg_set:
                        alg_index = np.find(alg_set == alg)
                        eval('acc_recorder(alg_index,iteration,theta_index) = sum(predict_' + str(alg) +'== y_true_te)/max(np.shape(y_true_te))')

                    index_search_high = predict_search != hidden_class_id
                    acc_recorder[max(np.shape(alg_set)) + 1][iteration][theta_index] = sum(predict_search[index_search_high] == y_true_te[index_search_high]) / max(np.shape(y_true_te[index_search_high]))
                    ME_feature_pool[iteration] = UA_feature_index
                    UA_feature_pool[iteration] = ME_feature_index
                    theta_index = theta_index + 1

                print('\n|')

            savepath = ['./Result/Exp_Mfeat/'];
            if(not os.path.exists(savepath, 'file')):
                os.mkdir(savepath)

            scipy.io.savemat(savepath + data + '_budget' + str(100 * budget_rio),
                             {'acc_recorder':acc_recorder,
                              'best_recorder':best_recorder,
                              'alg_set':alg_set,
                              'theta_set':theta_set,
                              'budget_rio_set':budget_rio_set,
                              'acc_min':acc_min})


if __name__ == '__main__':
    main()