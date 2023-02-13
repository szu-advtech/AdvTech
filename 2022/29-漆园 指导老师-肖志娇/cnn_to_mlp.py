from tensorflow.keras.models import load_model
from tensorflow.contrib.keras.api.keras.models import Sequential
from tensorflow.contrib.keras.api.keras.layers import Dense, Activation, Flatten, Conv2D, Lambda
from tensorflow.contrib.keras.api.keras.callbacks import LambdaCallback
from tensorflow.contrib.keras.api.keras.optimizers import SGD, Adam
from tensorflow.contrib.keras.api.keras import backend as K

import numpy as np
from setup_mnist import MNIST
import tensorflow as tf


def get_weights(file_name, inp_shape=(28,28,1)):
    model = load_model(file_name, custom_objects={'fn':fn, 'tf':tf})
    temp_weights = [layer.get_weights() for layer in model.layers]
    new_params = []
    eq_weights = []
    cur_size = inp_shape
    for p in temp_weights:
        if len(p) > 0:
            W, b = p
            eq_weights.append([])
            if len(W.shape) == 2:
                eq_weights.append([W, b])
            else:
                new_size = (cur_size[0]-W.shape[0]+1, cur_size[1]-W.shape[1]+1, W.shape[-1])
                flat_inp = np.prod(cur_size)
                flat_out = np.prod(new_size)
                new_params.append(flat_out)
                W_flat = np.zeros((flat_inp, flat_out))
                b_flat = np.zeros((flat_out))
                m,n,p = cur_size
                d,e,f = new_size
                for x in range(d):
                    for y in range(e):
                        for z in range(f):
                            b_flat[e*f*x+f*y+z] = b[z]
                            for k in range(p):
                                for idx0 in range(W.shape[0]):
                                    for idx1 in range(W.shape[1]):
                                        i = idx0 + x
                                        j = idx1 + y
                                        W_flat[n*p*i+p*j+k, e*f*x+f*y+z]=W[idx0, idx1, k, z]
                eq_weights.append([W_flat, b_flat])
                cur_size = new_size
    return eq_weights, new_params

def fn(correct, predicted):
        return tf.nn.softmax_cross_entropy_with_logits(labels=correct,
                                                       logits=predicted)

def convert(file_name, new_name, cifar = False):
    if not cifar:
        eq_weights, new_params = get_weights(file_name)
        data = MNIST()
    else:
        eq_weights, new_params = get_weights(file_name, inp_shape = (32,32,3))
        data = CIFAR()
    model = Sequential()
    model.add(Flatten(input_shape=data.train_data.shape[1:]))
    for param in new_params:
        model.add(Dense(param))
        model.add(Lambda(lambda x: tf.nn.relu(x)))
    model.add(Dense(10))
    
    for i in range(len(eq_weights)):
        try:
            print(eq_weights[i][0].shape)
        except:
            pass
        model.layers[i].set_weights(eq_weights[i])

    sgd = SGD(lr=0.01, decay=1e-5, momentum=0.9, nesterov=True)
    
    model.compile(loss=fn,
                  optimizer=sgd,
                  metrics=['accuracy'])

    model.save(new_name)
    acc = model.evaluate(data.validation_data, data.validation_labels)[1]

    return acc

if __name__ == '__main__':
        convert('models/mnist_cnn_4layer_5_3', 'models/mnist_cnn_as_mlp_4layer_5_3')