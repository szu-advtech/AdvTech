import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
from copy import deepcopy

class GCForest:
    """docstring for GCForest"""
    def __init__(self,windowRatio=[],  data_type="serial",scan_type="each",cascade_test_size=0.2, n_cascadeRF=4, n_cascadeRFtree=60, cascade_layer=10,min_sample_scan=0.07, min_sample_cascade=0.04, tolerance=0.0, n_jobs=-1):
        self.data_type=data_type
        self.scan_type=scan_type
        self.min_sample_scan=min_sample_scan
        self.windowRatio = windowRatio
        self.cascade_test_size = cascade_test_size
        self.n_cascadeRF = n_cascadeRF
        self.n_cascadeRFtree = n_cascadeRFtree
        self.cascade_layer = cascade_layer
        self.min_sample_cascade = min_sample_cascade
        self.tolerance = tolerance
        self.n_jobs = n_jobs

    def multi_grain_scan(self,X,y=None):

        self.levels_dataset = []

        rf = RandomForestClassifier(n_estimators=self.n_cascadeRFtree, max_features='sqrt',
                                    min_samples_split=self.min_sample_scan, n_jobs=self.n_jobs)
        erf = ExtraTreesClassifier(n_estimators=self.n_cascadeRFtree, min_samples_split=self.min_sample_scan,
                                   n_jobs=self.n_jobs)

        if y is not None:
            # 版本1：对每个扫出来的窗口样本集都各自训练森林。
            if self.scan_type=="each" :
                for i in range(len(self.windowRatio)):
                    print("windows:{}".format(i))
                    new_dataset = np.array([])

                    window_size = math.ceil(self.windowRatio[i] * X.shape[1])
                    for j in range(X.shape[1] - window_size + 1):
                        if self.data_type=="image" :
                            for k in range(X.shape[1] - window_size + 1):
                                win_image=X[:,j:j + window_size,k:k + window_size]
                                X_this_win=win_image.reshape(X.shape[0],-1)
                                rf.fit(X_this_win, y)
                                erf.fit(X_this_win, y)
                                #取的地方也改改
                                setattr(self, '_scantreerf_grain{}_no.{}.{}'.format(i, j,k), deepcopy(rf))
                                setattr(self, '_scantreeerf_grain{}_no.{}.{}'.format(i, j,k), deepcopy(erf))
                                pre_rf = rf.predict_proba(X_this_win)
                                if j == 0 & k == 0:
                                    new_dataset = pre_rf
                                else:
                                    new_dataset = np.concatenate((new_dataset, pre_rf), axis=1)

                                pre_erf = erf.predict_proba(X_this_win)
                                new_dataset = np.concatenate((new_dataset, pre_erf), axis=1)
                        elif self.data_type=="serial" :
                            X_this_win=X[:, j:j + window_size]
                            rf.fit(X_this_win, y)
                            erf.fit(X_this_win, y)
                            setattr(self, '_scantreerf_grain{}_no.{}'.format(i, j), deepcopy(rf))
                            setattr(self, '_scantreeerf_grain{}_no.{}'.format(i, j), deepcopy(erf))
                            pre_rf = rf.predict_proba(X_this_win)
                            if j == 0:
                                new_dataset = pre_rf
                            else:
                                new_dataset = np.concatenate((new_dataset, pre_rf), axis=1)

                            pre_erf = erf.predict_proba(X_this_win)
                            new_dataset = np.concatenate((new_dataset, pre_erf), axis=1)


                    self.levels_dataset.append(new_dataset)

            #纵向拼接扫出来的窗口样本集，再一起训练一个森林。
            elif self.scan_type=="all" :
                for i in range(len(self.windowRatio)):
                    print("windows:{}".format(i))
                    new_dataset = np.array([])
                    window_size = math.ceil(self.windowRatio[i] * X.shape[1])
                    x_to_train_forest=np.array([])
                    y_to_train_forest = np.array([])
                    for j in range(X.shape[1] - window_size + 1):
                        if self.data_type=="image" :
                            for k in range(X.shape[1] - window_size + 1):
                                win_image=X[:,j:j + window_size,k:k + window_size]
                                X_this_win=win_image.reshape(X.shape[0],-1)
                                if j == 0 & k==0:
                                    x_to_train_forest = X_this_win
                                    y_to_train_forest = y
                                else:
                                    x_to_train_forest = np.concatenate((x_to_train_forest, X_this_win), axis=0)
                                    y_to_train_forest = np.concatenate((y_to_train_forest, y), axis=0)
                        elif self.data_type=="serial" :
                            X_this_win=X[:, j:j + window_size]
                            if j == 0:
                                x_to_train_forest = X_this_win
                                y_to_train_forest = y
                            else:
                                x_to_train_forest = np.concatenate((x_to_train_forest, X_this_win), axis=0)
                                y_to_train_forest = np.concatenate((y_to_train_forest, y), axis=0)

                    rf.fit(x_to_train_forest,y_to_train_forest)
                    erf.fit(x_to_train_forest, y_to_train_forest)
                    setattr(self, '_scantreerf_grain{}'.format(i), deepcopy(rf))
                    setattr(self, '_scantreeerf_grain{}'.format(i), deepcopy(erf))
                    new_dataset_rf=np.array([])
                    new_dataset_erf = np.array([])
                    for j in range(X.shape[1] - window_size + 1):

                        if self.data_type=="image" :
                            for k in range(X.shape[1] - window_size + 1):
                                win_image = X[:, j:j + window_size, k:k + window_size]
                                X_this_win = win_image.reshape(X.shape[0], -1)
                                if j == 0 & k==0:
                                    new_dataset_rf = rf.predict_proba(X_this_win)
                                    new_dataset_erf = erf.predict_proba(X_this_win)
                                else:
                                    new_dataset_rf = np.concatenate((new_dataset_rf, rf.predict_proba(X_this_win)),axis=1)
                                    new_dataset_erf = np.concatenate((new_dataset_erf, erf.predict_proba(X_this_win)),axis=1)
                        elif self.data_type=="serial" :
                            X_this_win=X[:, j:j + window_size]
                            if j == 0:
                                new_dataset_rf = rf.predict_proba(X_this_win)
                                new_dataset_erf = erf.predict_proba(X_this_win)
                            else:
                                new_dataset_rf = np.concatenate((new_dataset_rf, rf.predict_proba(X_this_win)), axis=1)
                                new_dataset_erf = np.concatenate((new_dataset_erf, erf.predict_proba(X_this_win)), axis=1)
                    new_dataset = np.concatenate((new_dataset_rf, new_dataset_erf), axis=1)
                    self.levels_dataset.append(new_dataset)




        elif y is None:
            if self.scan_type == "each":
                for i in range(len(self.windowRatio)):
                    print("windows:{}".format(i))
                    new_dataset = np.array([])
                    window_size = math.ceil(self.windowRatio[i] * X.shape[1])
                    for j in range(X.shape[1] - window_size + 1):
                        if self.data_type=="image" :
                            for k in range(X.shape[1] - window_size + 1):
                                rf = getattr(self, '_scantreerf_grain{}_no.{}.{}'.format(i, j,k))
                                erf = getattr(self, '_scantreeerf_grain{}_no.{}.{}'.format(i, j,k))
                                win_image = X[:, j:j + window_size, k:k + window_size]
                                X_this_win = win_image.reshape(X.shape[0], -1)
                                pre_rf = rf.predict_proba(X_this_win)
                                if j == 0 & k == 0:
                                    new_dataset = pre_rf
                                else:
                                    new_dataset = np.concatenate((new_dataset, pre_rf), axis=1)

                                pre_erf = erf.predict_proba(X_this_win)
                                new_dataset = np.concatenate((new_dataset, pre_erf), axis=1)
                        elif self.data_type=="serial" :
                            rf = getattr(self, '_scantreerf_grain{}_no.{}'.format(i, j))
                            erf = getattr(self, '_scantreeerf_grain{}_no.{}'.format(i, j))
                            X_this_win = X[:, j:j + window_size]
                            pre_rf = rf.predict_proba(X_this_win)
                            if j == 0:
                                new_dataset = pre_rf
                            else:
                                new_dataset = np.concatenate((new_dataset, pre_rf), axis=1)

                            pre_erf = erf.predict_proba(X_this_win)
                            new_dataset = np.concatenate((new_dataset, pre_erf), axis=1)
                    self.levels_dataset.append(new_dataset)
            elif self.scan_type=="all" :
                for i in range(len(self.windowRatio)):
                    print("windows:{}".format(i))
                    new_dataset = np.array([])
                    window_size = math.ceil(self.windowRatio[i] * X.shape[1])
                    rf = getattr(self, '_scantreerf_grain{}'.format(i))
                    erf = getattr(self, '_scantreeerf_grain{}'.format(i))
                    new_dataset_rf = np.array([])
                    new_dataset_erf = np.array([])
                    for j in range(X.shape[1] - window_size + 1):
                        if self.data_type=="image" :
                            for k in range(X.shape[1] - window_size + 1):
                                win_image = X[:, j:j + window_size, k:k + window_size]
                                X_this_win = win_image.reshape(X.shape[0], -1)
                                if j == 0 & k==0:
                                    new_dataset_rf = rf.predict_proba(X_this_win)
                                    new_dataset_erf = erf.predict_proba(X_this_win)
                                else:
                                    new_dataset_rf = np.concatenate((new_dataset_rf, rf.predict_proba(X_this_win)),
                                                                    axis=1)
                                    new_dataset_erf = np.concatenate((new_dataset_erf, erf.predict_proba(X_this_win)),
                                                                     axis=1)
                        elif self.data_type=="serial" :
                            X_this_win=X[:, j:j + window_size]
                            if j == 0:
                                new_dataset_rf = rf.predict_proba(X_this_win)
                                new_dataset_erf = erf.predict_proba(X_this_win)
                            else:
                                new_dataset_rf = np.concatenate((new_dataset_rf, rf.predict_proba(X_this_win)),axis=1)
                                new_dataset_erf = np.concatenate((new_dataset_erf, erf.predict_proba(X_this_win)), axis=1)

                    new_dataset = np.concatenate((new_dataset_rf, new_dataset_erf), axis=1)
                    self.levels_dataset.append(new_dataset)




    def fit(self, X, y):
        self.multi_grain_scan(X,y)
        self.cascade_forest(self.levels_dataset,y)

    def predict_proba(self,X):
        self.multi_grain_scan(X)
        cascade_all_pred_prob = self.cascade_forest(self.levels_dataset)
        predict_proba = np.mean(cascade_all_pred_prob, axis=0)
        return predict_proba

    def predict(self,X):
        pred_prob = self.predict_proba(X)
        return np.argmax(pred_prob, axis=1) # may be changed to weight sum


    def cascade_forest(self, levels_dataset, y=None):
        if y is not None:
            # train
            self.n_layer = 0
            # split train and valid sets
            tol = self.tolerance
            split_per = self.cascade_test_size
            max_layers = self.cascade_layer


            X_train=[]
            X_test=[]
            y_train=[]
            y_test=[]
            for i in range(len(self.windowRatio)):
                X_train_item, X_test_item, y_train_item, y_test_item = train_test_split(levels_dataset[i],y, test_size=split_per,random_state=1)
                X_train.append(X_train_item)
                X_test.append(X_test_item)
                y_train.append(y_train_item)
                y_test.append(y_test_item)
            rf_erf_pred_ref = self._cascade_layer(X_train[0], y_train[0], level=0)

            self.n_layer += 1
            for m in range(len(self.windowRatio)):
                X_this_level = self._create_feat_arr(X_train[m], rf_erf_pred_ref)
                y_this_level = y_train[m]
                rf_erf_pred_ref = self._cascade_layer(X_this_level, y_this_level,level=m)

            accuracy_ref = self._cascade_evaluation(X_test, y_test)
            print(accuracy_ref)

            self.n_layer += 1
            #rf_erf_pred_layer = self._cascade_layer(feat_arr, y_train)
            for m in range(len(self.windowRatio)):
                if(m==0):
                    X_this_level=self._create_feat_arr(X_train[m], rf_erf_pred_ref)
                    y_this_level=y_train[m]
                else:
                    X_this_level = self._create_feat_arr(X_train[m], rf_erf_pred_layer)
                    y_this_level = y_train[m]
                rf_erf_pred_layer = self._cascade_layer(X_this_level, y_this_level,level=m)
            #问题：前后两次的误差总是一样。导致总是只有一层。说明其中肯定有问题。
            accuracy_layer = self._cascade_evaluation(X_test, y_test)
            print(accuracy_layer)

            while accuracy_layer > (accuracy_ref+tol) and self.n_layer <= max_layers:
                accuracy_ref = accuracy_layer
                rf_erf_pred_ref = rf_erf_pred_layer
                self.n_layer += 1
                for m in range(len(self.windowRatio)):
                    if (m == 0):
                        X_this_level = self._create_feat_arr(X_train[m], rf_erf_pred_ref)
                        y_this_level = y_train[m]
                    else:
                        X_this_level = self._create_feat_arr(X_train[m], rf_erf_pred_layer)
                        y_this_level = y_train[m]
                    rf_erf_pred_layer = self._cascade_layer(X_this_level, y_this_level, level=m)
                accuracy_layer = self._cascade_evaluation(X_test, y_test)
                print(accuracy_layer)
            if accuracy_layer <= accuracy_ref:

                n_cascadeRF = getattr(self, 'n_cascadeRF')
                for irf in range(n_cascadeRF):
                    for l in range(len(self.windowRatio)):
                        delattr(self, '_casrf{}_{}_{}'.format(self.n_layer, l,irf))
                        delattr(self, '_caserf{}_{}_{}'.format(self.n_layer, l,irf))
                self.n_layer -= 1
                print("total_layer:{}".format(self.n_layer))


        elif y is None:
            at_layer = 0
            rf_erf_pred_ref = self._cascade_layer(X=levels_dataset[0], layer=at_layer, level=0)
            at_layer += 1
            for n in range(len(self.windowRatio)):
                x_this_level=self._create_feat_arr(levels_dataset[n], rf_erf_pred_ref)
                rf_erf_pred_ref = self._cascade_layer(X=x_this_level, layer=at_layer,level=n)

            while at_layer < getattr(self, 'n_layer'):
                at_layer += 1
                for n in range(len(self.windowRatio)):
                    x_this_level = self._create_feat_arr(levels_dataset[n], rf_erf_pred_ref)
                    rf_erf_pred_ref = self._cascade_layer(X=x_this_level, layer=at_layer, level=n)

        return rf_erf_pred_ref


    def _cascade_layer(self, X, y=None, layer=0,level=0):

        n_tree = getattr(self, 'n_cascadeRFtree')
        n_cascadeRF = getattr(self, 'n_cascadeRF')
        min_samples = getattr(self, 'min_sample_cascade')

        n_jobs = getattr(self, 'n_jobs')
        rf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                     min_samples_split=self.min_sample_scan,  n_jobs=n_jobs)
        erf = ExtraTreesClassifier(n_estimators=n_tree,min_samples_split=self.min_sample_scan,n_jobs=n_jobs)

        rf_erf_pred = []
        if y is not None:
            #print('Adding/Training Layer, n_layer={}'.format(self.n_layer))
            for irf in range(n_cascadeRF):
                rf.fit(X, y)
                erf.fit(X, y)
                setattr(self, '_casrf{}_{}_{}'.format(self.n_layer,level, irf), deepcopy(rf))
                setattr(self, '_caserf{}_{}_{}'.format(self.n_layer,level, irf), deepcopy(erf))

                rf_erf_pred.append(rf.predict_proba(X))
                rf_erf_pred.append(erf.predict_proba(X))
        elif y is None:
            for irf in range(n_cascadeRF):
                rf = getattr(self, '_casrf{}_{}_{}'.format(layer,level, irf))
                erf = getattr(self, '_caserf{}_{}_{}'.format(layer,level, irf))
                rf_erf_pred.append(rf.predict_proba(X))
                rf_erf_pred.append(erf.predict_proba(X))

        return rf_erf_pred


    def _cascade_evaluation(self, X_test, y_test):
        casc_pred_prob = np.mean(self.cascade_forest(X_test), axis=0)
        casc_pred = np.argmax(casc_pred_prob, axis=1)
        casc_accuracy = accuracy_score(y_true=y_test[0], y_pred=casc_pred)
        return casc_accuracy

    def _create_feat_arr(self, X, prf_crf_pred):
        swap_pred = np.swapaxes(prf_crf_pred, 0, 1)
        add_feat = swap_pred.reshape([np.shape(X)[0], -1])
        feat_arr = np.concatenate([add_feat, X], axis=1)
        return feat_arr





