import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''
教学评价模块
'''


def evalute(score):
    read_data = np.loadtxt("score_list.txt")
    data = np.asarray(read_data)
    data = np.reshape(data, (-1, 1))
    Scare = MinMaxScaler(feature_range=(0, 1)).fit(data)
    emotion_score = Scare.transform(np.asarray(score).reshape((-1, 1)))
    return emotion_score
