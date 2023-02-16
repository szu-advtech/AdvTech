import itertools
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from copy import deepcopy

class GCForest:
    """docstring for GCForest"""
    def __init__(self, shape_1X=None, n_mgsRFtree=30, window=None, stride=1, cascade_test_size=0.2, n_cascadeRF=2, n_cascadeRFtree=101, cascade_layer=5, min_sample_mgs=0.1, min_sample_cascade=0.05, tolerance=0.0, n_jobs=1, mg_flag=True):

        self.shape_1X = shape_1X
        self.n_mgsRFtree = n_mgsRFtree
        self.window = window
        self.stride = stride
        self.cascade_test_size = cascade_test_size
        self.n_cascadeRF = n_cascadeRF
        self.n_cascadeRFtree = n_cascadeRFtree
        self.cascade_layer = cascade_layer
        self.min_sample_mgs = min_sample_mgs
        self.min_sample_cascade = min_sample_cascade
        self.tolerance = tolerance
        self.n_jobs = n_jobs
        self.mg_flag = mg_flag

    def fit(self, X, y):
        if self.mg_flag:
            X = self.mg_scanning(X,y)
        self.cascade_forest(X,y)

    def predict_proba(self,X):
        if self.mg_flag:
            X = self.mg_scanning(X)
        cascade_all_pred_prob = self.cascade_forest(X)
        predict_proba = np.mean(cascade_all_pred_prob, axis=0)
        return predict_proba

    def predict(self,X):
        pred_prob = self.predict_proba(X)
        return np.argmax(pred_prob, axis=1) # may be changed to weight sum

    def mg_scanning(self, X, y=None):
        # I do not want to implement it first
        return X

    def cascade_forest(self, X, y=None):
        if y is not None:
            # train
            self.n_layer = 0
            # split train and valid sets
            tol = self.tolerance
            split_per = self.cascade_test_size
            max_layers = self.cascade_layer

            X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split_per)
            self.n_layer += 1
            prf_crf_pred_ref = self._cascade_layer(X_train, y_train)
            accuracy_ref = self._cascade_evaluation(X_test, y_test)
            feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)

            self.n_layer += 1
            prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
            accuracy_layer = self._cascade_evaluation(X_test, y_test)
            while accuracy_layer > (accuracy_ref+tol) and self.n_layer <= max_layers:
                accuracy_ref = accuracy_layer
                prf_crf_pred_ref = prf_crf_pred_layer
                feat_arr = self._create_feat_arr(X_train, prf_crf_pred_ref)
                self.n_layer += 1
                prf_crf_pred_layer = self._cascade_layer(feat_arr, y_train)
                accuracy_layer = self._cascade_evaluation(X_test, y_test)

            if accuracy_layer < accuracy_ref:
                n_cascadeRF = getattr(self, 'n_cascadeRF')
                for irf in range(n_cascadeRF):
                    delattr(self, '_casprf{}_{}'.format(self.n_layer, irf))
                    delattr(self, '_cascrf{}_{}'.format(self.n_layer, irf))
                self.n_layer -= 1


        elif y is None:
            at_layer = 1
            prf_crf_pred_ref = self._cascade_layer(X, layer=at_layer)
            while at_layer < getattr(self, 'n_layer'):
                at_layer += 1
                feat_arr = self._create_feat_arr(X, prf_crf_pred_ref)
                print (feat_arr.shape)
                prf_crf_pred_ref = self._cascade_layer(feat_arr, layer=at_layer)

        return prf_crf_pred_ref


    def _cascade_layer(self, X, y=None, layer=0):

        n_tree = getattr(self, 'n_cascadeRFtree')
        n_cascadeRF = getattr(self, 'n_cascadeRF')
        min_samples = getattr(self, 'min_sample_cascade')

        n_jobs = getattr(self, 'n_jobs')
        prf = RandomForestClassifier(n_estimators=n_tree, max_features='sqrt',
                                     min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)
        crf = RandomForestClassifier(n_estimators=n_tree, max_features=1,
                                     min_samples_split=min_samples, oob_score=True, n_jobs=n_jobs)

        prf_crf_pred = []
        if y is not None:
            print('Adding/Training Layer, n_layer={}'.format(self.n_layer))
            for irf in range(n_cascadeRF):
                prf.fit(X, y)
                crf.fit(X, y)
                setattr(self, '_casprf{}_{}'.format(self.n_layer, irf), deepcopy(prf))
                setattr(self, '_cascrf{}_{}'.format(self.n_layer, irf), deepcopy(crf))
                prf_crf_pred.append(prf.oob_decision_function_)
                prf_crf_pred.append(crf.oob_decision_function_)
        elif y is None:
            for irf in range(n_cascadeRF):
                prf = getattr(self, '_casprf{}_{}'.format(layer, irf))
                crf = getattr(self, '_cascrf{}_{}'.format(layer, irf))
                prf_crf_pred.append(prf.predict_proba(X))
                prf_crf_pred.append(crf.predict_proba(X))

        return prf_crf_pred



    def _cascade_evaluation(self, X_test, y_test):
        casc_pred_prob = np.mean(self.cascade_forest(X_test), axis=0)
        casc_pred = np.argmax(casc_pred_prob, axis=1)
        casc_accuracy = accuracy_score(y_true=y_test, y_pred=casc_pred)
        return casc_accuracy

    def _create_feat_arr(self, X, prf_crf_pred):
        swap_pred = np.swapaxes(prf_crf_pred, 0, 1)
        add_feat = swap_pred.reshape([np.shape(X)[0], -1])
        feat_arr = np.concatenate([add_feat, X], axis=1)
        return feat_arr





