# this model is the Late Fusion.

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
import numpy as np
import pandas as pd

# 融合方法
def Late_Fusion(raw_result,HFF_result):
    final_result = (raw_result + HFF_result)/2
    return final_result

# 两个评价指标
def RMSE(y_predict, y_true):
    sum = 0
    n = len(y_predict)
    for i, j in zip(y_predict, y_true):
        sum = sum + (i - j) ** 2
        rmse = np.sqrt(sum / n)
    return rmse


def score(y_predict, y_true):
    sum = 0
    for i, j in zip(y_predict, y_true):
        z = i - j
        if z < 0:
            sum = sum + np.e ** (-z / 13) - 1
        else:
            sum = sum + np.e ** (z / 10) - 1
    return sum

# 网络模型
def fit_LSTM(trainX, trainY):

    # define LSTM
    model = Sequential()
    # 控制return_sequences=True/False, 运行看一下
    model.add(Bidirectional(LSTM(16, return_sequences=True), input_shape=(trainX.shape[1], trainX.shape[-1])))
    model.add(Bidirectional(LSTM(32, return_sequences=True), input_shape=(16, 1)))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['acc'])
    model.fit(trainX, trainY, epochs=50, batch_size=10, verbose=2)

    return model


GloUse_HFF = {}
GloUse_HFF['train_file'] = ['HFF/' + 'HFF_train_FD00' + str(i) for i in range(1, 5)]
GloUse_HFF['test_file'] = ['HFF/' + 'HFF_test_FD00' + str(i) for i in range(1, 5)]

GloUse_raw = {}
GloUse_raw['train_file'] = ['raw stream/' +'train_FD00' + str(i) for i in range(1, 5)]
GloUse_raw['test_file'] = ['raw stream/' +'test_FD00' + str(i) for i in range(1, 5)]


# GloUse['train_file'] = ['raw stream/' + 'train_FD00' + str(i) for i in range(1, 5)]

GloUse = {}
GloUse['RUL_file'] = ['RUL_FD00' + str(i) for i in range(1, 5)]
GloUse['SL'] = [14, 14, 14, 14]
#training and testing trajectories
GloUse['train_units'] = [100, 260, 100, 249]
GloUse['test_units'] = [100, 259, 100, 248]
n = 1


# 构建训练数据
df_train_HFF = pd.read_pickle(GloUse_HFF['train_file'][n - 1] + ".pickle")
train_HFF_input = df_train_HFF.iloc[:, :-2].values
train_HFF_output = df_train_HFF.iloc[:, -1].values
df_train_raw = pd.read_pickle(GloUse_raw['train_file'][n - 1] + ".pickle")
train_raw_input = df_train_raw.iloc[:, :-2].values
train_raw_output = df_train_raw.iloc[:, -1].values.reshape(-1, )
# 构建测试数据

df_test_HFF = pd.read_pickle(GloUse_HFF['test_file'][n - 1] + ".pickle")
test_HFF_input = df_test_HFF.iloc[:, :-2].values
test_HFF_output = df_test_HFF.iloc[:, -1].values
df_test_raw = pd.read_pickle(GloUse_raw['test_file'][n - 1] + ".pickle")
test_raw_input = df_test_raw.iloc[:, :-2].values
test_raw_output = df_test_raw.iloc[:, -1].values


train_HFF_input = train_HFF_input.reshape((train_HFF_input.shape[0], 1, train_HFF_input.shape[1]))
train_HFF_output = train_HFF_output
train_raw_input = train_raw_input.reshape((train_raw_input.shape[0], 1, train_raw_input.shape[1]))
train_raw_output = train_raw_output
test_HFF_input = test_HFF_input.reshape((test_HFF_input.shape[0], 1, test_HFF_input.shape[1]))
test_HFF_output = test_HFF_output
test_raw_input = test_raw_input.reshape((test_raw_input.shape[0], 1, test_raw_input.shape[1]))
test_raw_output = test_raw_output

# raw
model_raw = fit_LSTM(train_raw_input, train_raw_output)
# 对训练数据的Y进行预测
train_raw_Predict = model_raw.predict(train_raw_input)
# 对测试数据的Y进行预测
test_raw_Predict = model_raw.predict(test_raw_input)
# 计算指标
train_raw_rmse = RMSE(train_raw_Predict, train_raw_output)
test_raw_rmse = RMSE(test_raw_Predict, test_raw_output)
#score
train_raw_score = score(train_raw_Predict, train_raw_output)
test_raw_score = score(test_raw_Predict, test_raw_output)


# HFF
model_HFF = fit_LSTM(train_HFF_input, train_HFF_output)
# 对训练数据的Y进行预测
train_HFF_Predict = model_HFF.predict(train_HFF_input)
# 对测试数据的Y进行预测
test_HFF_Predict = model_HFF.predict(test_HFF_input)

# 计算指标
#RMSE
train_HFF_rmse = RMSE(train_HFF_Predict, train_HFF_output)
test_HFF_rmse = RMSE(test_HFF_Predict, test_HFF_output)
#score
train_HFF_score = score(train_HFF_Predict, train_HFF_output)
test_HFF_score = score(test_HFF_Predict, test_HFF_output)

# 结果融合
trainPredict = Late_Fusion(train_raw_Predict, train_HFF_Predict)
testPredict = Late_Fusion(test_raw_Predict, test_HFF_Predict)
# 计算指标
#RMSE
train_rmse = RMSE(trainPredict, train_HFF_output)
test_rmse = RMSE(testPredict, test_HFF_output)
#score
train_score = score(trainPredict, train_HFF_output)
test_score = score(testPredict, test_HFF_output)

