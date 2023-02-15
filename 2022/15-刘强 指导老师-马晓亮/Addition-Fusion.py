# this model is the addition Fusion.


from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import  Dropout
from keras.layers import Add
from keras.layers import LSTM
from keras.layers import Bidirectional
import numpy as np
import pandas as pd


# evaluation methods
def RMSE(y_predict, y_true):
    sum = 0
    n = len(y_predict)
    for i, j in zip(y_predict, y_true):
        sum = sum + (i - j) ** 2
    return sum / n


def score(y_predict, y_true):
    sum = 0
    for i, j in zip(y_predict, y_true):
        z = i - j
        if z < 0:
            sum = sum + np.e ** (-z / 13) - 1
        else:
            sum = sum + np.e ** (z / 10) - 1
    return sum

# load data
def HFF_data(n):
    GloUse_HFF = {}
    GloUse_HFF['train_file'] = ['HFF/' + 'HFF_train_FD00' + str(i) for i in range(1, 5)]
    GloUse_HFF['test_file'] = ['HFF/' + 'HFF_test_FD00' + str(i) for i in range(1, 5)]
    GloUse = {}
    GloUse['RUL_file'] = ['RUL_FD00' + str(i) for i in range(1, 5)]
    GloUse['SL'] = [14, 14, 14, 14]
    # training and testing trajectories
    GloUse['train_units'] = [100, 260, 100, 249]
    GloUse['test_units'] = [100, 259, 100, 248]
    # 构建训练数据
    df_train_HFF = pd.read_pickle(GloUse_HFF['train_file'][n - 1] + ".pickle")
    train_HFF_input = df_train_HFF.iloc[:, :-2].values
    train_HFF_output = df_train_HFF.iloc[:, -1].values
    # 构建测试数据

    df_test_HFF = pd.read_pickle(GloUse_HFF['test_file'][n - 1] + ".pickle")
    test_HFF_input = df_test_HFF.iloc[:, :-2].values
    test_HFF_output = df_test_HFF.iloc[:, -1].values

    # reshape
    train_HFF_input = train_HFF_input.reshape((train_HFF_input.shape[0], 1, train_HFF_input.shape[1]))
    train_HFF_output = train_HFF_output
    test_HFF_input = test_HFF_input.reshape((test_HFF_input.shape[0], 1, test_HFF_input.shape[1]))
    test_HFF_output = test_HFF_output


    return train_HFF_input, train_HFF_output, test_HFF_input, test_HFF_output


def raw_data(n):
    GloUse_raw = {}
    GloUse_raw['train_file'] = ['raw stream/' + 'train_FD00' + str(i) for i in range(1, 5)]
    GloUse_raw['test_file'] = ['raw stream/' + 'test_FD00' + str(i) for i in range(1, 5)]
    GloUse = {}
    GloUse['RUL_file'] = ['RUL_FD00' + str(i) for i in range(1, 5)]
    GloUse['SL'] = [14, 14, 14, 14]
    # training and testing trajectories
    GloUse['train_units'] = [100, 260, 100, 249]
    GloUse['test_units'] = [100, 259, 100, 248]

    # 构建训练数据
    df_train_raw = pd.read_pickle(GloUse_raw['train_file'][n - 1] + ".pickle")
    train_raw_input = df_train_raw.iloc[:, :-2].values
    train_raw_output = df_train_raw.iloc[:, -1].values.reshape(-1, )
    # 构建测试数据
    df_test_raw = pd.read_pickle(GloUse_raw['test_file'][n - 1] + ".pickle")
    test_raw_input = df_test_raw.iloc[:, :-2].values
    test_raw_output = df_test_raw.iloc[:, -1].values
    #reshape
    train_raw_input = train_raw_input.reshape((train_raw_input.shape[0], 1, train_raw_input.shape[1]))
    train_raw_output = train_raw_output
    test_raw_input = test_raw_input.reshape((test_raw_input.shape[0], 1, test_raw_input.shape[1]))
    test_raw_output = test_raw_output

    return train_raw_input, train_raw_output, test_raw_input, test_raw_output

# model
def fit_LSTM(trainX_raw, trainX_HFF, trainY):

    # raw
    input_raw = Input (shape=(trainX_raw.shape[1], trainX_raw.shape[-1])) #这个输入维度需要修改
    x_raw_1 = Bidirectional(LSTM(16, return_sequences=True), input_shape=(trainX_raw.shape[1], trainX_raw.shape[-1]))(input_raw)
    x_raw_2 = Bidirectional(LSTM(32, return_sequences=True), input_shape=(16, 1))(x_raw_1)
    x_raw_3 = Dense(16, activation='relu')(x_raw_2)
    x_raw_4 = Dropout(0.2)(x_raw_3)

    #HFF
    input_HFF = Input (shape=(trainX_HFF.shape[1], trainX_HFF.shape[-1])) #这个输入维度需要修改
    x_HFF_1 = Bidirectional(LSTM(16, return_sequences=True), input_shape=(trainX_HFF.shape[1], trainX_HFF.shape[-1]))(input_HFF)
    x_HFF_2 = Bidirectional(LSTM(32, return_sequences=True), input_shape=(16, 1))(x_HFF_1)
    x_HFF_3 = Dense(16, activation='relu')(x_HFF_2)
    x_HFF_4 = Dropout(0.2)(x_HFF_3)

    # fusion
    added = Add()([x_raw_4,x_HFF_4])
    out_1 = Dense(8, activation='relu')(added)
    out_2 = Dropout(0.2)(out_1)
    out_3 = Dense(1, activation='relu')(out_2)
    out = Dropout(0.2)(out_3)
    model = Model(inputs = [input_raw, input_HFF], outputs = out)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit([trainX_raw, trainX_HFF], trainY, epochs=50, batch_size=10, verbose=2)

    return model

#load data
n = 1 # Change the data set
train_raw_input, train_raw_output, test_raw_input, test_raw_output = raw_data(n)
train_HFF_input, train_HFF_output, test_HFF_input, test_HFF_output = HFF_data(n)

# train
model = fit_LSTM(train_raw_input, train_HFF_input, train_raw_output)
# 对训练数据的Y进行预测
train_Predict = model.predict([train_raw_input, train_HFF_input])
# 对测试数据的Y进行预测
test_Predict = model.predict([test_raw_input, test_HFF_input])
# 计算指标
train_rmse = RMSE(train_Predict, train_raw_output)
test_rmse = RMSE(test_Predict, test_raw_output)
#score
train_score = score(train_Predict, train_raw_output)
test_score = score(test_Predict, test_raw_output)
