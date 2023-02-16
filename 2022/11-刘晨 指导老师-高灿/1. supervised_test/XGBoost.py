import numpy as np
import pandas as pd

import transtab
import xgboost as xgb
from sklearn.metrics import roc_auc_score

allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
    = transtab.load_data('credit-g')

params = {
    'booster': 'gbtree',
    'n_estimators': 50,
    # 'objective': 'multi:softmax',
    'objective': 'binary:logistic',
    'num_class': 2,
    'gamma': 0.1,
    'max_depth': 4,
    'lambda': 2,
    'subsample': 0.7,
    'colsample_bytree': 0.75,
    'min_child_weight': 3,
    'silent': 0,
    'eta': 0.1,
    'seed': 123,
    'nthread': 4,
}

num_round = 100

x_train = pd.get_dummies(trainset[0])
y_train = trainset[1]

dtrain = xgb.DMatrix(x_train, y_train)
dtest = xgb.DMatrix(testset[0])

model = xgb.train(params, dtrain, num_round)

y_pred = model.predict(dtest)

roc_score = roc_auc_score(np.array(testset[1]), y_pred)
print(roc_score)

