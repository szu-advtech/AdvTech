from .BaseNet import *


def Generator(config=None):
    if config.back_net == 'cnn':
        return Feature_CNN(config)
    elif config.back_net == 'convlstmv2':
        return Feature_ConvLSTMv2(config)


def Disentangler(config=None):
    return Feature_disentangle(config)


def Reconstructor(config=None):
    return Reconstructor_Net(config)


def Mine(config=None):
    return Mine_Net(config)


def Classifier(config=None):
    return Predictor_Net(config)
