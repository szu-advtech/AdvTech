import pandas as pd

import transtab

# load a WorldHappiness_Corruption_2015_2020 and start vanilla supervised training
# allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data(['credit-g', 'credit-approval'])

allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data('credit-approval')

# build transtab classifier model
model = transtab.build_classifier(cat_cols, num_cols, bin_cols)

# start training
training_arguments = {
    'num_epoch': 50,
    'eval_metric': 'val_loss',
    'eval_less_is_better': True,
    'output_dir': './checkpoint',
    'batch_size': 128,
    'lr': 5e-5,
    'weight_decay': 1e-4,
    }

# x_train: pd.DataFrame = trainset[0]
# y_train: pd.Series = trainset[1]
# n = x_train.shape[0] / 3 * 2
# x_train1, y_train1 = x_train.iloc[0:n], y_train.iloc[0:n]
# x_train2, y_train2 = x_train.iloc[-n:-1], y_train.iloc[-n:-1]

# transtab.train(model, trainset, valset, **training_arguments)
# model.save('./ckpt/pretrained')


# allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = transtab.load_data('credit-approval')

model.load('./ckpt/pretrained')

model.update({'cat': cat_cols, 'num': num_cols, 'bin': bin_cols, 'num_class': 2})

training_arguments = {
    # 'num_epoch': 50,
    'eval_metric': 'val_loss',
    'eval_less_is_better': True,
    'output_dir': './checkpoint',
    'batch_size': 128,
    'lr': 5e-5,
    'weight_decay': 1e-4,
    }

model.train(model, trainset, valset, **training_arguments)

x_test = testset[0]
y_test = testset[1]
ypred = transtab.predict(model, x_test)

transtab.predict(model, x_test)

transtab.evaluate(ypred, y_test, seed=123, metric='auc')

