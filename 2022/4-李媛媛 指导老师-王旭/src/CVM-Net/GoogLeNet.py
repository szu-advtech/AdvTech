import tensorflow as tf


class GoogleNet(object):
    def __init__(self):
        with tf.variable_scope("net", initializer=tf.random_normal_initializer(0.0, 0.001)):
            with tf.variable_scope("Input"):
                input_x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
                net = tf.image.resize_images(images=input_x, size=(224, 224))

            net = self.__conv2d('conv1', net, 64, 7, 2)
            net = self.__max_pool('pool2', net, 3, 2)
            net = self.__conv2d('conv3', net, 192, 3, 1)
            net = self.__conv2d('conv4', net, 192, 3, 1)
            net = self.__max_pool('pool5', net, 3, 2)
            net = self.__inception('inception6', net, 64, 96, 128, 16, 32, 32)
            net = self.__inception('inception7', net, 64, 96, 128, 16, 32, 32)
            net = self.__inception('inception8', net, 128, 128, 192, 32, 96, 64)
            net = self.__inception('inception9', net, 128, 128, 192, 32, 96, 64)
            net = self.__max_pool('pool10', net, 3, 2)
            net = self.__inception('inception11', net, 192, 96, 208, 16, 48, 64)
            net = self.__inception('inception12', net, 192, 96, 208, 16, 48, 64)
            net = self.__inception('inception13', net, 160, 112, 224, 24, 64, 64)
            net = self.__inception('inception14', net, 160, 112, 224, 24, 64, 64)
            net = self.__inception('inception15', net, 128, 128, 256, 24, 64, 64)
            net = self.__inception('inception16', net, 128, 128, 256, 24, 64, 64)
            net = self.__inception('inception17', net, 112, 144, 288, 32, 64, 64)
            net = self.__inception('inception18', net, 112, 144, 288, 32, 64, 64)
            net = self.__inception('inception19', net, 256, 160, 320, 32, 128, 128)
            net = self.__inception('inception20', net, 256, 160, 320, 32, 128, 128)
            net = self.__max_pool('pool21', net, 3, 2)
            net = self.__inception('inception22', net, 256, 160, 320, 32, 128, 128)
            net = self.__inception('inception23', net, 256, 160, 320, 32, 128, 128)
            net = self.__inception('inception24', net, 384, 192, 384, 48, 128, 128)
            net = self.__inception('inception25', net, 384, 192, 384, 48, 128, 128)
            net = self.__avg_pool('pool26', net, 7, 1, padding='VALID')
            net = tf.nn.dropout(net, keep_prob=0.4)
            shape = net.get_shape()
            net = tf.reshape(net, shape=[-1, shape[1] * shape[2] * shape[3]])
            net = self.__fc('fc27', net, 10)
            logits = tf.nn.softmax(net)

            self.logits = logits

    def __fc(self, name, net, units, with_activation=True):
        with tf.variable_scope(name):
            input_units = net.get_shape()[-1]
            w = tf.get_variable('w', shape=[input_units, units])
            b = tf.get_variable('b', shape=[units])
            net = tf.add(tf.matmul(net, w), b)
            if with_activation:
                net = tf.nn.relu(net)
            return net

    def __conv2d(self, name, net, output_channels, window_size, stride_size, padding='SAME', with_activation=True):
        with tf.variable_scope(name):
            input_channels = net.get_shape()[-1]
            filter = tf.get_variable('w', shape=[window_size, window_size, input_channels, output_channels])
            bias = tf.get_variable('b', shape=[output_channels])
            net = tf.nn.bias_add(tf.nn.conv2d(input=net, filter=filter,
                                              strides=[1, stride_size, stride_size, 1],
                                              padding=padding), bias)
            if with_activation:
                net = tf.nn.relu(net)
            return net

    def __max_pool(self, name, net, window_size, stride_size, padding='SAME'):
        with tf.variable_scope(name):
            net = tf.nn.max_pool(net, ksize=[1, window_size, window_size, 1],
                                 strides=[1, stride_size, stride_size, 1],
                                 padding=padding)
            return net

    def __avg_pool(self, name, net, window_size, stride_size, padding='SAME'):
        with tf.variable_scope(name):
            net = tf.nn.avg_pool(net, ksize=[1, window_size, window_size, 1],
                                 strides=[1, stride_size, stride_size, 1],
                                 padding=padding)
            return net

    def __inception(self, name, net,
                    branch1_output_channels,
                    branch2_reduce_output_channels, branch2_output_channels,
                    branch3_reduce_output_channels, branch3_output_channels,
                    branch4_output_channels):
        with tf.variable_scope(name):
            with tf.variable_scope("branch_1"):
                net1 = self.__conv2d('conv1', net, branch1_output_channels, 1, 1)
            with tf.variable_scope("branch_2"):
                tmp_net = self.__conv2d('conv1', net, branch2_reduce_output_channels, 1, 1)
                net2 = self.__conv2d('conv2', tmp_net, branch2_output_channels, 3, 1)
            with tf.variable_scope("branch_3"):
                tmp_net = self.__conv2d('conv1', net, branch3_reduce_output_channels, 1, 1)
                net3 = self.__conv2d('conv2', tmp_net, branch3_output_channels, 5, 1)
            with tf.variable_scope("branch_4"):
                tmp_net = self.__max_pool('pool1', net, 3, 1)
                net4 = self.__conv2d('conv1', tmp_net, branch4_output_channels, 1, 1)
            with tf.variable_scope("Concat"):
                net = tf.concat([net1, net2, net3, net4], axis=-1)
            return net
