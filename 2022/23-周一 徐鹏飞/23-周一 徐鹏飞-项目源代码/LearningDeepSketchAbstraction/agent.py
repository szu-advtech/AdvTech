import random

import abs_utils
import dataset
import rl_config
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras import models
from keras import layers


def discnt_rwd(R):
    discnt_r = []
    for b in range(rl_config.batch_size()):
        sum_reward = 0.
        dis_r = np.zeros_like(R, dtype=np.float32)
        for t in reversed(range(len(R))):
            sum_reward = R[t] + rl_config.gamma() * sum_reward
            dis_r[t] = sum_reward
        discnt_r.append(dis_r)

    return discnt_r


class Agent(object):
    def __init__(self):
        self.ss_idx = tf.Variable(0, dtype=tf.int32)

        inputs = layers.Input(shape=(dataset.find_max_points_num(), 3))
        x = layers.Bidirectional(layers.GRU(units=128, dropout=0.5), merge_mode='concat')(inputs)
        x = layers.Dropout(0.5)(x)

        # issue: units of fc unspecified
        # 2 accounts for forward / backward in bgru
        x = layers.Dense(256)(x[2 * 5 * self.ss_idx.value()
                                :2 * 5 * (self.ss_idx.value() + 1)])
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256)(x)
        x = layers.Dropout(0.5)(x)

        x = layers.Dense(256)(x)
        x = layers.Dropout(0.5)(x)

        outputs = layers.Dense(2, activation='softmax', use_bias=True)(x)
        self.model = models.Model(inputs=inputs, outputs=outputs)
        self.opt = tf.keras.optimizers.Adam(learning_rate=rl_config.lr())

        self.model.compile(optimizer=self.opt)
        print("Agent model:")
        print(self.model.summary())
        return

    def select_action(self, sketch_np, ss_idx: int):
        if np.isnan(sketch_np).any():
            raise AssertionError("Nan")

        self.ss_idx.assign(ss_idx)
        prob = self.model(np.array([sketch_np]))
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        action = dist.sample()
        action = int(action.numpy()[0])
        return action

    def fit(self, S_batch, A_batch, R_batch):
        assert len(S_batch) == len(A_batch) == len(R_batch)

        total_steps: int = sum([len(S) for S in S_batch])

        # discounted reward
        discnt_R_batch = []
        for b in range(rl_config.batch_size()):
            R = R_batch[b]
            sum_reward = 0.
            discnt_r = np.zeros_like(R, dtype=np.float32)
            for t in reversed(range(len(R))):
                sum_reward = R[t] + rl_config.gamma() * sum_reward
                discnt_r[t] = sum_reward
            discnt_R_batch.append(discnt_r)

        Ss, As, Rs = [], [], []
        ss_indices = []
        for S, A, R in zip(S_batch, A_batch, discnt_R_batch):
            Ss.extend(S)
            As.extend(A)
            Rs.extend(R)
            ss_indices.extend([i for i in range(len(S))])

        random.seed(1122211)
        random.shuffle(Ss)
        random.seed(1122211)
        random.shuffle(As)
        random.seed(1122211)
        random.shuffle(Rs)
        random.seed(1122211)
        random.shuffle(ss_indices)

        act_probs = []
        losses = []
        for state, reward, action, ss_idx in zip(Ss, Rs, As, ss_indices):
            self.ss_idx.assign(ss_idx)
            with tf.GradientTape() as tape:
                action_prob = self.model(np.array([state]))
                action_prob = action_prob[0]
                dist = tfp.distributions.Categorical(probs=action_prob, allow_nan_stats=False, dtype=tf.float32)
                log_prob = dist.log_prob(action)
                loss = -log_prob * reward
            # print(loss[0])
            # print(action_prob)
            act_probs.append(action_prob)
            losses.append(loss)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return act_probs, losses

    def observe(self, state):
        raise NotImplementedError()

    def evaluate(self, state, action):
        raise NotImplementedError()
