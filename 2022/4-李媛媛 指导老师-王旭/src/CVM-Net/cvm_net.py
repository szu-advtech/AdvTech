
from ResNet import Resnet
from GoogLeNet import GoogleNet
import loupe as lp
from siamese_fc import Siamese_FC
from transnet_v2 import TransNet

import tensorflow as tf

def cvm_net_I(x_sat, x_grd, keep_prob, trainable):
    with tf.device('/gpu:3'):
        GoogleNet_grd = GoogleNet()
        grd_local = GoogleNet_grd.__conv2d(x_grd, keep_prob, trainable, 'GoogleNet_grd')
        with tf.variable_scope('netvlad_grd'):
            netvlad_grd = lp.NetVLAD(feature_size=512, max_samples=tf.shape(grd_local)[1] * tf.shape(grd_local)[2],
                                     cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
                                     is_training=trainable)
            grd_vlad = netvlad_grd.forward(grd_local)

        GoogleNet_sat = GoogleNet()
        sat_local = GoogleNet_sat.__conv2d(x_sat, keep_prob, trainable, 'GoogleNet_sat')
        with tf.variable_scope('netvlad_sat'):
            netvlad_sat = lp.NetVLAD(feature_size=512, max_samples=tf.shape(sat_local)[1] * tf.shape(sat_local)[2],
                                     cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
                                     is_training=trainable)
            sat_vlad = netvlad_sat.forward(sat_local)

    with tf.device('/gpu:1'):
        fc = Siamese_FC()
        sat_global, grd_global = fc.siamese_fc(sat_vlad, grd_vlad, trainable, 'dim_reduction')

    return sat_global, grd_global


def cvm_net_II(x_sat, x_grd, keep_prob, trainable):
    with tf.device('/gpu:3'):
        GoogleNet_grd = GoogleNet()
        grd_local = GoogleNet_grd.__conv2d(x_grd, keep_prob, trainable, 'GoogleNet_grd')

        GoogleNet_sat = GoogleNet()
        sat_local = GoogleNet_sat.__conv2d(x_sat, keep_prob, trainable, 'GoogleNet_sat')

        transnet = TransNet()
        trans_sat, trans_grd = transnet.transform(sat_local, grd_local, keep_prob, trainable,
                                                  'transformation')

        with tf.variable_scope('netvlad') as scope:
            netvlad_sat = lp.NetVLAD(feature_size=512, max_samples=tf.shape(trans_sat)[1] * tf.shape(trans_sat)[2],
                                     cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
                                     is_training=trainable)
            sat_global = netvlad_sat.forward(trans_sat, True)

            scope.reuse_variables()

            netvlad_grd = lp.NetVLAD(feature_size=512, max_samples=tf.shape(trans_grd)[1] * tf.shape(trans_grd)[2],
                                     cluster_size=64, output_dim=4096, gating=True, add_batch_norm=False,
                                     is_training=trainable)
            grd_global = netvlad_grd.forward(trans_grd, True)

    return sat_global, grd_global