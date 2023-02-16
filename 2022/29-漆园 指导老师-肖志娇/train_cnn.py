import numpy as np
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten, Lambda, Conv2D, BatchNormalization
from tensorflow.contrib.keras.api.keras.models import load_model
from tensorflow.contrib.keras.api.keras import backend as K
from tensorflow.contrib.keras.api.keras.optimizers import Adam

import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
import os

def train(data, file_name, filters, kernels, num_epochs=50, batch_size=128, train_temp=1, init=None, activation=tf.nn.relu, bn = False):
    model = Sequential()
    model.add(Conv2D(filters[0], kernels[0], input_shape=data.train_data.shape[1:]))
    if bn:
        model.add(BatchNormalization())
    model.add(Lambda(activation))
    for f, k in zip(filters[1:], kernels[1:]):
        model.add(Conv2D(f,k))
        if bn:
            model.add(BatchNormalization())
        model.add(Lambda(activation))
    model.add(Flatten())
    model.add(Dense(10))

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
    print("Traing a {} layer model, saving to {}".format(len(filters) + 1, file_name))
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
    train(MNIST(), file_name="models/mnist_cnn_4layer_5_3", filters=[5,5,5], kernels = [3,3,3], num_epochs=10)
    train(MNIST(), file_name="models/mnist_cnn_4layer_5_3_sigmoid", filters=[5,5,5], kernels = [3,3,3], num_epochs=10, activation = tf.sigmoid)
    train(MNIST(), file_name="models/mnist_cnn_4layer_5_3_tanh", filters=[5,5,5], kernels = [3,3,3], num_epochs=10, activation = tf.tanh)
    train(MNIST(), file_name="models/mnist_cnn_4layer_5_3_atan", filters=[5,5,5], kernels = [3,3,3], num_epochs=10, activation = tf.atan)

    
