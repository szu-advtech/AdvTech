import numpy as np
from tensorflow.contrib.keras.api.keras.models import Sequential, Model
from tensorflow.contrib.keras.api.keras.layers import Input, Dense, Activation, Flatten, Lambda, Conv2D, Add, AveragePooling2D, BatchNormalization, Lambda
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.optimizers import SGD, Adam
from tensorflow.python.keras.engine.base_layer import Layer

import tensorflow as tf
from setup_mnist import MNIST
import os


class ResidualStart(Layer):
    def __init__(self, **kwargs):
        super(ResidualStart, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ResidualStart, self).build(input_shape)

    def call(self, x):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

class ResidualStart2(Layer):
    def __init__(self, **kwargs):
        super(ResidualStart2, self).__init__(**kwargs)

    def build(self, input_shape):
        super(ResidualStart2, self).build(input_shape)

    def call(self, x):
        return x

    def compute_output_shape(self, input_shape):
        return input_shape

def Residual(f, activation):
    def res(x):
        x = ResidualStart()(x)
        x1 = Conv2D(f, 3, strides=1, padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = Lambda(activation)(x1)
        x1 = Conv2D(f, 3, strides=1, padding='same')(x1)
        x1 = BatchNormalization()(x1)
        return Add()([x1, x])
    return res
   
def Residual2(f, activation):
    def res(x):
        x = ResidualStart2()(x)
        x1 = Conv2D(f, 3, strides=2, padding='same')(x)
        x1 = BatchNormalization()(x1)
        x1 = Lambda(activation)(x1)
        x1 = Conv2D(f, 3, strides=1, padding='same')(x1)
        x1 = BatchNormalization()(x1)
        x2 = Conv2D(f, 3, strides=2, padding='same')(x)
        x2 = BatchNormalization()(x2)
        return Add()([x1, x2])
    return res

def train(data, file_name, nlayer, num_epochs=10, batch_size=128, train_temp=1, init=None, activation=tf.nn.relu):
    """
    Train a n-layer CNN for MNIST and CIFAR
    """
    inputs = Input(shape=(28,28,1))
    if nlayer == 2:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(10)(x)
    if nlayer == 3:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(10)(x)
    if nlayer == 4:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = Residual(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(10)(x)
    if nlayer == 5:
        x = Residual2(8, activation)(inputs)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual(8, activation)(x)
        x = Lambda(activation)(x)
        x = Residual2(16, activation)(x)
        x = Lambda(activation)(x)
        x = Residual(16, activation)(x)
        x = Lambda(activation)(x)
        x = AveragePooling2D(pool_size=7)(x)
        x = Flatten()(x)
        x = Dense(10)(x)

    model = Model(inputs=inputs, outputs=x)

    if init != None:
        model.load_weights(init)

    def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted/train_temp)
    sgd = Adam()    

    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])
    
    model.summary()

    history = model.fit(data.train_data, data.train_labels,
              batch_size=batch_size,
              validation_data=(data.validation_data, data.validation_labels),
              epochs=num_epochs,
              shuffle=True)

    if file_name != None:
        model.save(file_name)
    
    return {'model':model, 'history':history}

if not os.path.isdir('models'):
    os.makedirs('models')


if __name__ == '__main__':
    train(MNIST(), file_name="models/mnist_resnet_2", nlayer=2, activation = tf.nn.relu)
    train(MNIST(), file_name="models/mnist_resnet_2_sigmoid", nlayer=2, activation = tf.sigmoid)
    train(MNIST(), file_name="models/mnist_resnet_2_tanh", nlayer=2, activation = tf.tanh)
    train(MNIST(), file_name="models/mnist_resnet_2_atan", nlayer=2, activation = tf.atan)
