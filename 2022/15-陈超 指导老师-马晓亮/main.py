from self_paced_ensemble.canonical_ensemble import *
from self_paced_ensemble.utils import load_covtype_dataset
from self_paced_ensemble.self_paced_ensemble.base import sort_dict_by_key
from self_paced_ensemble import SelfPacedEnsembleClassifier
from dynamic_ensemble import DynamicEnsemble
from imblearn.ensemble import EasyEnsembleClassifier
import pandas as pd
from time import time
from collections import Counter
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score


RANDOM_STATE = 2022

'''
四个数据集：creditcard，covtype， kddcup，payment simulation， 其中 kddcup 有 dos vs. prb 和 # dos vs. r2l 两种数据集

'''


# creditcard
# data = pd.read_csv('data/creditcard.csv')
# feature_col = [col for col in data.columns if col not in ['Time', 'Class']]
# X, y = data[feature_col], data['Class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

# covtype dataset
X_train, X_test, y_train, y_test = load_covtype_dataset(subset=0.1, random_state=RANDOM_STATE)

# kddcup
# names = [col for col in range(41)]
# names.append('class')
# df = pd.read_csv('data/kddcup.data', names=names)
# dos = ['smurf.', 'back.', 'land.', 'neptune.', 'pod.',  'teardrop.']
# prb = ['ipsweep.', 'nmap.', 'satan.', 'portsweep.']
# r2l = ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'spy.', 'phf.', 'warezclient.', 'warezmaster.']
# u2r = ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']

# dos vs. prb

# df_dos = df.loc[df['class'].isin(dos)].reset_index(drop=True)
# df_dos['class'] = 0
# df_prb = df.loc[df['class'].isin(prb)].reset_index(drop=True)
# df_prb['class'] = 1
# df = pd.concat([df_dos, df_prb], axis=0).reset_index(drop=True)
# feature_col = [col for col in df.columns if col not in [1, 2, 3, 'class']]
# X, y = df[feature_col], df['class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# dos vs. r2l

# df_dos = df.loc[df['class'].isin(dos)].reset_index(drop=True)
# df_dos['class'] = 0
# df_r2l = df.loc[df['class'].isin(r2l)].reset_index(drop=True)
# df_r2l['class'] = 1
# df = pd.concat([df_dos, df_r2l], axis=0).reset_index(drop=True)
# feature_col = [col for col in df.columns if col not in [1, 2, 3, 'class']]
# X, y = df[feature_col], df['class']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)

# Payment Simulation
# data = pd.read_csv('data/PS_20174392719_1491204439457_log.csv')
# feature_col = [col for col in data.columns if col not in ['type', 'isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest']]
# X, y = data[feature_col], data['isFraud']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)



origin_distr = sort_dict_by_key(Counter(y_train))
test_distr = sort_dict_by_key(Counter(y_test))
print('Original training dataset shape %s' % origin_distr)
print('Original test dataset shape     %s' % test_distr)


init_kwargs = {
    'n_estimators': 10,
    'random_state': RANDOM_STATE,
}
fit_kwargs = {
    'X': X_train,
    'y': y_train,
}

ensembles = {
    'DynamicEnsembleClassifier': DynamicEnsemble,
    'SelfPacedEnsembleClassifier': SelfPacedEnsembleClassifier,
    # 'SMOTEBagging': SMOTEBaggingClassifier,
    # 'SMOTEBoost': SMOTEBoostClassifier,
    # 'UnderBagging': UnderBaggingClassifier,
    # 'RUSBoost': RUSBoostClassifier,
    # 'BalanceCascade': BalanceCascadeClassifier,
    # 'EasyEnsemble': EasyEnsembleClassifier
}

fit_ensembles = {}
for ensemble_name, ensemble_class in ensembles.items():
    ensemble_clf = ensemble_class(**init_kwargs)
    print('Training {:^20s} '.format(ensemble_name), end='')
    start_time = time()
    ensemble_clf.fit(X_train, y_train)
    fit_time = time() - start_time
    y_pred = ensemble_clf.predict_proba(X_test)[:, 1]
    score = average_precision_score(y_test, y_pred)
    print ('| Average precision score {:.3f} | Time {:.3f}s'.format(score, fit_time))
