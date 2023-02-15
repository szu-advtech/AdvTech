import torch
from lightgbm import LGBMClassifier

import datasetloader
from all_models import GraphCDA, GraphCDALast, GraphCDANoGAT, MyModel1
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import lightgbm as lgb
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
import evaluation
import time
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

def main(dataset_path,model,clfs,n_fold,lr,epoch):
    kf = KFold(n_splits=n_fold,shuffle=True)
    f = open("../result/GraphCDA _result.txt", "a")
    f.write('time:\t' + str(time.asctime(time.localtime(time.time()))) + "\n")
    f.write(f"dataset={str(dataset_path)}\n,model={str(model.__class__.__name__)},lr={lr},epoch={epoch}" + "\n")
    dataset = datasetloader.get_GCN_dataset(*dataset_path)
    cd_pairs = dataset['cd_pairs']

    acc = [list() for i in range(len(clfs))]
    prec = [list() for i in range(len(clfs))]
    sens = [list() for i in range(len(clfs))]
    f1_score = [list() for i in range(len(clfs))]
    mcc = [list() for i in range(len(clfs))]
    auc = [list() for i in range(len(clfs))]
    auprc = [list() for i in range(len(clfs))]
    for train_index, test_index in kf.split(cd_pairs):
        # 获得5折后的cd矩阵，训练和测试的数对
        cd_matix, train_cd_pairs, test_cd_pairs = datasetloader.get_cdmatrix(cd_pairs, train_index, test_index)
        # 将cd矩阵加入到模型数据
        dataset["cd_matix"] = cd_matix
        # 获得训练后的cd关联分数，c特征，d特征
        score, cir_fea, dis_fea = datasetloader.feature_representation(model, dataset, lr, epoch)

        # 获取训练和测试的数据特征
        train_dataset = datasetloader.get_clf_dataset(cir_fea, dis_fea, train_cd_pairs)
        test_dataset = datasetloader.get_clf_dataset(cir_fea, dis_fea, test_cd_pairs)

        X_train, y_train = train_dataset[:, :-2], train_dataset[:, -2:]
        X_test, y_test = test_dataset[:, :-2], test_dataset[:, -2:]


        for i,clf in enumerate(clfs):
            clf.fit(X_train, y_train[:, 0])
            y_pred = clf.predict(X_test)
            y_prob = clf.predict_proba(X_test)



            acc1, prec1, sens1, f1_score1, MCC1, AUC1, AUPRC1 = evaluation.calculate_performace(len(y_pred),y_pred,y_prob[:, 1],y_test[:, 0])

            # f.write(f'{acc}\t{prec}\t{sens}\t{f1_score}\t{MCC}\t{AUC}\t{AUPRC}\n')
            acc[i].append(acc1)
            prec[i].append(prec1)
            sens[i].append(sens1)
            f1_score[i].append(f1_score1)
            mcc[i].append(MCC1)
            auc[i].append(AUC1)
            auprc[i].append(AUPRC1)

    for i,clf in enumerate(clfs):
        ave_acc = round(sum(acc[i]) /n_fold,4)
        ave_prec  = round(sum(prec[i]) /n_fold,4)
        ave_sens  = round(sum(sens[i]) /n_fold,4)
        ave_f1_score  = round(sum(f1_score[i]) /n_fold,4)
        ave_mcc  = round(sum(mcc[i]) /n_fold,4)
        ave_auc  = round(sum(auc[i]) /n_fold,4)
        ave_auprc  = round(sum(auprc[i]) /n_fold,4)
        print(clf)
        print('Acc\tprec\tsens\tf1_score\tMCC\tAUC\tAUPRC')
        print(f'{ave_acc }\t{ave_prec }\t{ave_sens }\t{ave_f1_score }\t{ave_mcc }\t{ave_auc }\t{ave_auprc }')
        f.write(f'{clf}\n{ave_acc }\t{ave_prec }\t{ave_sens }\t{ave_f1_score }\t{ave_mcc }\t{ave_auc }\t{ave_auprc }\n')
    f.close()


if __name__ == '__main__':
    models=[GraphCDANoGAT(),GraphCDA(4),GraphCDALast(),MyModel1()]
    clfs=[RandomForestClassifier(n_estimators=200, n_jobs=11, max_depth=20),DecisionTreeClassifier(),
          SVC(probability=True),GaussianNB(),LogisticRegression(max_iter=1000),LGBMClassifier(max_depth=20, learning_rate=0.05, objective='binary',n_estimators=200,n_jobs=11)]
    dataset_paths=[["../data/c_d.csv","../data/d_d.csv","../data/c_c.csv"],
                  ["../data/c_d.csv","../data/dss.csv","../data/cfs.csv"],
                  ["../data/c_d.csv","../data/dgs.csv","../data/cgs.csv"]]


    for i in range(10):
        main(dataset_paths[0], GraphCDA(4),[RandomForestClassifier(),LGBMClassifier()], 5, 0.005, 400)




