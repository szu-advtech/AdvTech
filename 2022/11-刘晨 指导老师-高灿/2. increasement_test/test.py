import transtab
import pandas as pd

# pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
# conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
# pip --default-timeout=1000 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

# load WorldHappiness_Corruption_2015_2020 by specifying WorldHappiness_Corruption_2015_2020 name
# allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
#      = transtab.load_data('dresses-sales')

training_arguments = {
    'num_epoch': 100,
    'batch_size': 64,
    'lr': 1e-4,
    'eval_metric': 'val_loss',
    'eval_less_is_better': True,
    'output_dir': './checkpoint'
    }


allset, trainset, valset, testset, cat_cols, num_cols, bin_cols \
    = transtab.load_data('liver-disorders')
# build classifier
model = transtab.build_classifier(cat_cols, num_cols, bin_cols, )

x_train: pd.DataFrame = trainset[0]
y_train: pd.Series = trainset[1]
x_train1, y_train1 = x_train.iloc[0:2000], y_train.iloc[0:2000]
x_train2, y_train2 = x_train.iloc[0:5000], y_train.iloc[0:5000]

# start training
transtab.train(model, trainset, valset, **training_arguments)


x_test = testset[0]
y_test = testset[1]
ypred = transtab.predict(model, x_test)

transtab.predict(model, x_test)

transtab.evaluate(ypred, y_test, seed=123, metric='auc')

