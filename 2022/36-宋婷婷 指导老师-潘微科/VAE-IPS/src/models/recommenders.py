"""
Recommender models used for the semi-synthetic experiments
in the paper "Unbiased Recommender Learning from Missing-Not-At-Random Implicit Feedback".
"""
from __future__ import absolute_import, print_function

from abc import ABCMeta, abstractmethod

import numpy as np
import tensorflow as tf


class AbstractRecommender(metaclass=ABCMeta):
    """Abstract base class for evaluator class."""

    @abstractmethod
    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        raise NotImplementedError()

    @abstractmethod
    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        raise NotImplementedError()

    @abstractmethod
    def create_losses(self) -> None:
        """Create the losses."""
        raise NotImplementedError()

    @abstractmethod
    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        raise NotImplementedError()


class vaeRecommender(AbstractRecommender):
    """Implicit Recommenders based on pointwise approach."""

    def __init__(self, num_items: np.array, dim: int, eta: float, lam: float, model_name: str) -> None:
        """Initialize Class."""
        self.num_items = num_items
        self.dim = dim
        self.eta = eta
        self.lam = lam
        self.model_name = model_name

        # Build the graphs
        self.create_placeholders()
        self.build_graph()
        self.create_losses()
        self.add_optimizer()

    def create_placeholders(self) -> None:
        """Create the placeholders to be used."""
        self.input_ph = tf.placeholder(tf.float32, [None, self.num_items], name='input_placeholder')
        self.scores = tf.placeholder(tf.float32, [None, self.num_items], name='score_placeholder')

        self.keep_prob_ph = tf.placeholder_with_default(1.0, shape=None)
        self.anneal_ph = tf.placeholder_with_default(1., shape=None)
        self.is_training_ph = tf.placeholder_with_default(0., shape=None)

    def build_graph(self) -> None:
        """Build the main tensorflow graph with embedding layers."""
        with tf.name_scope('embedding_layer'):
            # initialize user-item matrices and biases
            self.weights_q_mu = tf.get_variable('weights_q_mu', shape=[self.num_items, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())    # W_mu
            self.weights_q_std = tf.get_variable('weights_q_std', shape=[self.num_items, self.dim],
                                                   initializer=tf.contrib.layers.xavier_initializer())    # W_std
            self.weights_p = tf.get_variable('weights_p', shape=[self.dim, self.num_items],
                                                   initializer=tf.contrib.layers.xavier_initializer())    # V
            self.biases_q_mu = tf.get_variable('biases_q_mu', shape=[self.dim],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.001))    # biases_q_mu
            self.biases_q_std = tf.get_variable('biases_q_std', shape=[self.dim],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.001))    # biases_q_std
            self.biases_p = tf.get_variable('biases_p', shape=[self.num_items],
                                                   initializer=tf.truncated_normal_initializer(stddev=0.001))    # biases_p

        with tf.variable_scope('prediction'):
            _, self.logits, self.KL = self.forward_pass()    # X_batch_hat: shape=(?, num_items)
            self.preds = tf.nn.log_softmax(self.logits)
            self.preds2 = tf.nn.log_softmax(1. - self.logits)

    def q_graph(self):  # encoder
        h = tf.nn.l2_normalize(self.input_ph, 1)    # shape=(?, num_items)
        h = tf.nn.dropout(h, self.keep_prob_ph)     # shape=(?, num_items)

        mu_q = tf.matmul(h, self.weights_q_mu) + self.biases_q_mu    # shape=(?, dim)
        logvar_q = tf.matmul(h, self.weights_q_std) + self.biases_q_std    # shape=(?, dim)

        std_q = tf.exp(0.5 * logvar_q)    # shape=(?, dim)
        KL = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar_q + mu_q**2 + tf.exp(logvar_q) - 1), axis=1))    # shape=()

        return mu_q, std_q, KL

    def p_graph(self, z):  # decoder
        h = tf.matmul(z, self.weights_p) + self.biases_p    # shape=(?, num_items)
        return h

    def forward_pass(self):
        # q-network
        mu_q, std_q, KL = self.q_graph()
        epsilon = tf.random_normal(tf.shape(std_q))    # shape=(?, dim)

        sampled_z = mu_q + self.is_training_ph * epsilon * std_q    # shape=(?, dim)

        # p-network
        logits = self.p_graph(sampled_z)    # shape=(?, num_items)

        return tf.train.Saver(), logits, KL

    def create_losses(self) -> None:
        """Create the losses."""
        with tf.name_scope('losses'):
            # define the unbiased binary cross entropy loss
            if self.model_name == 'vae':    # vae loss
                print('vae  loss')
                self.weighted_mse = -tf.reduce_mean(tf.reduce_sum(self.preds * self.input_ph, axis=1))
            else:    # vae-ips or vae-ips-imp loss
                print('vae-ips or vae-ips-imp  loss')
                self.weighted_mse = -tf.reduce_mean(tf.reduce_sum(self.preds * self.input_ph / self.scores, axis=1))


            # # add the L2-regularizer terms.
            reg_term_embeds = tf.nn.l2_loss(self.weights_q_mu) + tf.nn.l2_loss(self.weights_q_std) + tf.nn.l2_loss(self.weights_p)

            self.loss = self.weighted_mse + self.anneal_ph * self.KL + self.lam * reg_term_embeds

    def add_optimizer(self) -> None:
        """Add the required optimiser to the graph."""
        with tf.name_scope('optimizer'):
            # set Adam Optimizer.
            self.apply_grads = tf.train.AdamOptimizer(learning_rate=self.eta).minimize(self.loss)
