import keras.backend as K
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras import layers
from keras import utils as np_utils
import os
import tensorflow as tf
import scipy.io as sio
import numpy as np
import h5py
import scipy.io as sio


class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clustering')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# computing an auxiliary target distribution
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


def slices(x, index):
    return x[:, index]


def autoencoder_Conv2D(input_shape):
    input_img = Input(shape=input_shape)
    dim = input_shape[2]

    ######################################################################################################
    # Block 1
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='conv11')(input_img)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ######################################################################################################
    residual = Conv2D(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 2
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='conv21')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='conv22')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 2 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    ######################################################################################################
    # Fully Connected Layer
    encoded22 = GlobalAveragePooling2D(name='embedding22')(x)

    ######################################################################################################
    residual = Conv2D(96, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 3
    x = SeparableConv2D(96, (3, 3), padding='same', use_bias=False, name='conv31')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(96, (3, 3), padding='same', use_bias=False, name='conv32')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 3 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    ######################################################################################################
    # Fully Connected Layer
    encoded32 = GlobalAveragePooling2D(name='embedding32')(x)

    ######################################################################################################
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 4
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='conv41')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='conv42')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 4 Pool
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)
    x = layers.add([x, residual])

    ######################################################################################################
    # Block 6
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='conv61')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ######################################################################################################
    # Fully Connected Layer
    encoded62 = GlobalAveragePooling2D(name='embedding62')(x)

    ######################################################################################################
    wei22 = Dense(1, activation='relu')(encoded22)
    wei32 = Dense(1, activation='relu')(encoded32)
    wei62 = Dense(1, activation='relu')(encoded62)
    wei = layers.concatenate([wei22, wei32, wei62])
    wei = Dense(6, activation='relu')(wei)
    wei = Dense(3, activation='sigmoid')(wei)
    wei = Lambda(lambda x: x + 1)(wei)
    wei22 = Lambda(slices, arguments={"index": 0})(wei)
    wei32 = Lambda(slices, arguments={"index": 1})(wei)
    wei62 = Lambda(slices, arguments={"index": 2})(wei)
    encoded22 = Multiply()([encoded22, wei22])
    encoded32 = Multiply()([encoded32, wei32])
    encoded62 = Multiply()([encoded62, wei62])
    encoded = layers.concatenate([encoded62, encoded32, encoded22])

    ######################################################################################################
    # Block 6
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='deconv61')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    ######################################################################################################
    residual = Conv2DTranspose(96, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 4
    x = SeparableConv2D(96, (3, 3), padding='same', use_bias=False, name='deconv41')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(96, (3, 3), padding='same', use_bias=False, name='deconv42')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 4 Pool
    x = UpSampling2D((2, 2))(x)
    x = layers.add([x, residual])

    ######################################################################################################
    residual = Conv2DTranspose(64, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 3
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='deconv31')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(64, (3, 3), padding='same', use_bias=False, name='deconv32')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 3 Pool
    x = UpSampling2D((2, 2))(x)
    x = layers.add([x, residual])

    ######################################################################################################
    residual = Conv2DTranspose(dim, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    # Block 2
    x = SeparableConv2D(dim, (3, 3), padding='same', use_bias=False, name='deconv21')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = SeparableConv2D(dim, (3, 3), padding='same', use_bias=False, name='deconv22')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Block 2 Pool
    x = UpSampling2D((2, 2))(x)
    x = layers.add([x, residual])

    ######################################################################################################
    # Block 1
    decoded = SeparableConv2D(dim, (3, 3), padding='same', use_bias=False, name='deconv11')(x)

    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


# data generator function
def data_generator(data, targets, batch_sizee):
    batches = (len(data) + batch_sizee - 1) // batch_sizee
    while (True):
        for i in range(batches):
            X = data[i * batch_sizee: (i + 1) * batch_sizee]
            Y = targets[i * batch_sizee: (i + 1) * batch_sizee]
            yield (X, Y)


def run_pretrain():
    place = 'houston'
    n_clusters = 15

    # 使用第一张与第三张GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 按需动态分配显卡内存
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # zsyzsyzsy input shape
    fdata = h5py.File('mat_' + place + '_32.h5', 'r')
    # fdata = h5py.File('mat_' + place + '_64.h5', 'r')
    # fdata = h5py.File('mat_' + place + '_nsct_32.h5', 'r')
    # fdata = h5py.File('mat_' + place + '_nsct_64.h5', 'r')
    # f.keys()  # 可以查看所有的主键
    train_x = fdata['train_x']
    x_shape = train_x.shape
    train_y = np.random.randint(0, n_clusters, x_shape[0])
    # for test
    print('HDF5 file read finish~')
    print(train_x.shape)
    print(train_y.shape)

    # 预处理数据
    # zsyzsyzsy input shape
    input_shape = (x_shape[1], x_shape[2], x_shape[3])

    # convert class vectors to binary class matrices
    train_y = np_utils.to_categorical(train_y, n_clusters)
    # for test
    print(train_y.shape)

    # biuld the encoder and autoencoder model
    autoencoder, encoder = autoencoder_Conv2D(input_shape=input_shape)
    autoencoder.summary()

    # checkpoint
    filepath_checkpoint222 = "checkpoint_" + place + "_best_model_weights_32.hdf5"
    # filepath_checkpoint222 = "checkpoint_" + place + "_best_model_weights_64.hdf5"
    # filepath_checkpoint222 = "checkpoint_" + place + "_best_model_weights_nsct32.hdf5"
    # filepath_checkpoint222 = "checkpoint_" + place + "_best_model_weights_nsct64.hdf5"
    checkpoint222 = ModelCheckpoint(filepath_checkpoint222, monitor='loss', verbose=2,
                                    save_best_only=True, save_weights_only=True, mode='min', period=1)
    callbacks_list = [checkpoint222]

    # Pretrain covolutional autoencoder
    pretrain_epochs = 300
    batch_size = 128
    autoencoder.compile(optimizer='Nadam', loss='mse')
    autoencoder.fit_generator(generator=data_generator(train_x, train_x, batch_size),
                              steps_per_epoch=(len(train_x) + batch_size - 1) // batch_size,
                              epochs=pretrain_epochs, verbose=2, callbacks=callbacks_list)


def run_deep_cluster():
    place = 'houston'
    n_clusters = 15

    # 使用第一张与第三张GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 按需动态分配显卡内存
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # zsyzsyzsy input shape
    fdata = h5py.File('mat_' + place + '_32.h5', 'r')
    # fdata = h5py.File('mat_' + place + '_64.h5', 'r')
    # fdata = h5py.File('mat_' + place + '_nsct_32.h5', 'r')
    # fdata = h5py.File('mat_' + place + '_nsct_64.h5', 'r')
    # f.keys()  # 可以查看所有的主键
    train_x = fdata['train_x']
    x_shape = train_x.shape
    # zsyzsyzsy input shape
    input_shape = (x_shape[1], x_shape[2], x_shape[3])

    # build the multiple output model
    autoencoder, encoder = autoencoder_Conv2D(input_shape=input_shape)
    autoencoder.load_weights("checkpoint_" + place + "_best_model_weights_32.hdf5")
    # autoencoder.load_weights("checkpoint_" + place + "_best_model_weights_64.hdf5")
    # autoencoder.load_weights("checkpoint_" + place + "_best_model_weights_nsct32.hdf5")
    # autoencoder.load_weights("checkpoint_" + place + "_best_model_weights_nsct64.hdf5")

    # Build clustering model with convolutional autoencoder
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    model = Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
    model.compile(loss=['categorical_crossentropy', 'mse'], optimizer='Adam')

    # Initialize cluster centers using k-means
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder.predict(train_x))
    model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    y_pred_last = np.copy(y_pred)

    # deep cluster
    batch_size = 128
    loss = 0
    index = 0
    train_num = train_x.shape[0]
    maxiter = int(train_num / batch_size) * 300
    update_interval = int(train_num / batch_size)
    index_array = np.arange(train_num)
    tol = 0.001  # tolerance threshold to stop training
    flag = 1
    delta_flag = 666

    for ite in range(int(maxiter)):
        if ite % update_interval == 0:
            print('flag: %d **********************************************************' % flag)
            flag = flag + 1

            DCmodel = Model(inputs=model.input, outputs=model.get_layer(name='clustering').output)
            q = DCmodel.predict(train_x, verbose=0)
            p = target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)

            # check stop criterion
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            print('delta_label: %.5f' % delta_label)

            if ite > 0 and delta_label < delta_flag:
                delta_flag = delta_label
                model.save_weights('deep_cluster_' + place + '_best_model_weights_32.hdf5')
                # model.save_weights('deep_cluster_' + place + '_best_model_weights_64.hdf5')
                # model.save_weights('deep_cluster_' + place + '_best_model_weights_nsct32.hdf5')
                # model.save_weights('deep_cluster_' + place + '_best_model_weights_nsct64.hdf5')

            if ite > 0 and delta_label < tol:
                print('delta_label: %.5f < tol: %.5f' % (delta_label, tol))
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index + 1) * batch_size, train_x.shape[0])]
        loss = model.train_on_batch(x=train_x[idx], y=[p[idx], train_x[idx]])
        print('Iter %d: loss = %.5f & %.5f' % (ite, loss[0], loss[1]))
        index = index + 1 if (index + 1) * batch_size <= train_x.shape[0] else 0


def predict_embedding_vector():
    place = 'houston'
    img = sio.loadmat("Houston.mat")
    img = img['data']
    img_nsct = sio.loadmat("Houston_nsct.mat")
    img_nsct = img_nsct['nsct']

    # 使用第一张与第三张GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 按需动态分配显卡内存
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    # biuld the encoder and autoencoder model
    autoencoder_64, encoder_64 = autoencoder_Conv2D(input_shape=(64, 64, 144))
    autoencoder_64.load_weights('deep_cluster_' + place + '_best_model_weights_64.hdf5')
    autoencoder_64.compile(optimizer='Adam', loss='mse')

    autoencoder_32, encoder_32 = autoencoder_Conv2D(input_shape=(32, 32, 144))
    autoencoder_32.load_weights('deep_cluster_' + place + '_best_model_weights_32.hdf5')
    autoencoder_32.compile(optimizer='Adam', loss='mse')

    autoencoder_nsct64, encoder_nsct64 = autoencoder_Conv2D(input_shape=(64, 64, 48))
    autoencoder_nsct64.load_weights('deep_cluster_' + place + '_best_model_weights_nsct64.hdf5')
    autoencoder_nsct64.compile(optimizer='Adam', loss='mse')

    autoencoder_nsct32, encoder_nsct32 = autoencoder_Conv2D(input_shape=(32, 32, 48))
    autoencoder_nsct32.load_weights('deep_cluster_' + place + '_best_model_weights_nsct32.hdf5')
    autoencoder_nsct32.compile(optimizer='Adam', loss='mse')

    ################################################################################################
    sample_train = sio.loadmat("sample_train_" + place + ".mat")
    sample_train = sample_train["sample_train"]
    shape_train = sample_train.shape
    vector_train_64 = np.zeros((shape_train[0], 288))
    vector_train_32 = np.zeros((shape_train[0], 288))
    vector_train_nsct64 = np.zeros((shape_train[0], 288))
    vector_train_nsct32 = np.zeros((shape_train[0], 288))
    for ii in range(shape_train[0]):
        row_ii = sample_train[ii, 1]
        col_ii = sample_train[ii, 2]
        # for hsi64
        patch_ii = np.zeros((1, 64, 64, 144))
        patch_ii[0, :, :, :] = img[row_ii-32:row_ii+32, col_ii-32:col_ii+32, :]
        vector_train_64[ii, :] = encoder_64.predict(patch_ii, verbose=0)
        # for hsi32
        patch_ii = np.zeros((1, 32, 32, 144))
        patch_ii[0, :, :, :] = img[row_ii - 16:row_ii + 16, col_ii - 16:col_ii + 16, :]
        vector_train_32[ii, :] = encoder_32.predict(patch_ii, verbose=0)
        # for nsct64
        patch_ii = np.zeros((1, 64, 64, 48))
        patch_ii[0, :, :, :] = img_nsct[row_ii - 32:row_ii + 32, col_ii - 32:col_ii + 32, :]
        vector_train_nsct64[ii, :] = encoder_nsct64.predict(patch_ii, verbose=0)
        # for nsct32
        patch_ii = np.zeros((1, 32, 32, 48))
        patch_ii[0, :, :, :] = img_nsct[row_ii - 16:row_ii + 16, col_ii - 16:col_ii + 16, :]
        vector_train_nsct32[ii, :] = encoder_nsct32.predict(patch_ii, verbose=0)

    # save the mat file
    sio.savemat("hsi64_" + place + "_train_data.mat", {'train_data': vector_train_64})
    sio.savemat("hsi32_" + place + "_train_data.mat", {'train_data': vector_train_32})
    sio.savemat("nsct64_" + place + "_train_data.mat", {'train_data': vector_train_nsct64})
    sio.savemat("nsct32_" + place + "_train_data.mat", {'train_data': vector_train_nsct32})

    ################################################################################################
    sample_test = sio.loadmat("sample_test_" + place + ".mat")
    sample_test = sample_test["sample_test"]
    shape_test = sample_test.shape
    vector_test_64 = np.zeros((shape_test[0], 288))
    vector_test_32 = np.zeros((shape_test[0], 288))
    vector_test_nsct64 = np.zeros((shape_test[0], 288))
    vector_test_nsct32 = np.zeros((shape_test[0], 288))
    for ii in range(shape_test[0]):
        row_ii = sample_test[ii, 1]
        col_ii = sample_test[ii, 2]
        # for hsi64
        patch_ii = np.zeros((1, 64, 64, 144))
        patch_ii[0, :, :, :] = img[row_ii - 32:row_ii + 32, col_ii - 32:col_ii + 32, :]
        vector_test_64[ii, :] = encoder_64.predict(patch_ii, verbose=0)
        # for hsi32
        patch_ii = np.zeros((1, 32, 32, 144))
        patch_ii[0, :, :, :] = img[row_ii - 16:row_ii + 16, col_ii - 16:col_ii + 16, :]
        vector_test_32[ii, :] = encoder_32.predict(patch_ii, verbose=0)
        # for nsct64
        patch_ii = np.zeros((1, 64, 64, 48))
        patch_ii[0, :, :, :] = img_nsct[row_ii - 32:row_ii + 32, col_ii - 32:col_ii + 32, :]
        vector_test_nsct64[ii, :] = encoder_nsct64.predict(patch_ii, verbose=0)
        # for nsct32
        patch_ii = np.zeros((1, 32, 32, 48))
        patch_ii[0, :, :, :] = img_nsct[row_ii - 16:row_ii + 16, col_ii - 16:col_ii + 16, :]
        vector_test_nsct32[ii, :] = encoder_nsct32.predict(patch_ii, verbose=0)

    # save the mat file
    sio.savemat("hsi64_" + place + "_test_data.mat", {'test_data': vector_test_64})
    sio.savemat("hsi32_" + place + "_test_data.mat", {'test_data': vector_test_32})
    sio.savemat("nsct64_" + place + "_test_data.mat", {'test_data': vector_test_nsct64})
    sio.savemat("nsct32_" + place + "_test_data.mat", {'test_data': vector_test_nsct32})


def run_RF_UMsDFL():
    place = 'houston'

    # 使用第一张与第三张GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # 按需动态分配显卡内存
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # 不全部占满显存, 按需分配
    tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))

    ###################################################################################
    train_data = sio.loadmat("hsi64_"+place+"_train_data.mat")
    train_data = train_data["train_data"]
    train_label = sio.loadmat(place+"_train_label.mat")
    train_label = train_label["train_label"]
    train_label = train_label.ravel()

    tt_data = sio.loadmat("hsi32_"+place+"_train_data.mat")
    tt_data = tt_data["train_data"]
    train_data = np.hstack((train_data, tt_data))

    tt_data = sio.loadmat("nsct64_" + place + "_train_data.mat")
    tt_data = tt_data["train_data"]
    train_data = np.hstack((train_data, tt_data))

    tt_data = sio.loadmat("nsct32_" + place + "_train_data.mat")
    tt_data = tt_data["train_data"]
    train_data = np.hstack((train_data, tt_data))

    spectral_data = sio.loadmat("spectral_"+place+"_train_data.mat")
    spectral_data = spectral_data["train_data"]
    train_data = np.hstack((train_data, spectral_data))

    ###################################################################################
    model = RandomForestClassifier(n_estimators=300)
    model.fit(train_data, train_label)

    ###################################################################################
    test_data = sio.loadmat("hsi64_"+place+"_test_data.mat")
    test_data = test_data["test_data"]

    tt_data = sio.loadmat("hsi32_"+place+"_test_data.mat")
    tt_data = tt_data["test_data"]
    test_data = np.hstack((test_data, tt_data))

    tt_data = sio.loadmat("nsct64_" + place + "_test_data.mat")
    tt_data = tt_data["test_data"]
    test_data = np.hstack((test_data, tt_data))

    tt_data = sio.loadmat("nsct32_" + place + "_test_data.mat")
    tt_data = tt_data["test_data"]
    test_data = np.hstack((test_data, tt_data))

    spectral_data = sio.loadmat("spectral_"+place+"_test_data.mat")
    spectral_data = spectral_data["test_data"]
    test_data = np.hstack((test_data, spectral_data))

    predict_y = model.predict(test_data)
    predict_y = predict_y.reshape((predict_y.shape[0], 1))
    print(predict_y.shape)
    sio.savemat(place+"_test_RF_predictY.mat", {'predict_y': predict_y})
    print(place+"_test_RF_predictY.mat")


if __name__ == '__main__':
    run_pretrain()
    run_deep_cluster()
    predict_embedding_vector()
    run_RF_UMsDFL()





