import numpy as np
import matplotlib.pyplot as plt
import PIL.Image as Image
import random
import copy
import dataset
import rl_config
import tensorflow as tf
from classifier_config import classifier_model_path
from global_config import ss_len
from gym import Env, spaces
from tensorflow import keras
from keras import models
from abs_utils import debug_pause
from abs_utils import plot_sketch

selected_classifier_model_path: str = "./training/epoch_4_eacc89"
    # classifier_model_path() + "/on_auth_preprocessed/epoch_11_8987"


class SAEnv(Env):
    def render(self, mode="human"):
        pass

    def __init__(self, shared_classifier=None):
        super(SAEnv, self).__init__()

        # reward generator
        if shared_classifier is not None:
            self.__classifier__ = shared_classifier
        else:
            with tf.device("/gpu:1"):
                self.__classifier__: models.Model = keras.models.load_model(selected_classifier_model_path)

        self.__sketch_np__ = None
        self.__label_gt__ = -1
        self.__points_num__: int = -1

        # current stroke segment index
        self.__ss_idx__: int = 0

        # 0 - skip(remove), 1 - keep
        self.action_space = spaces.Discrete(2)

        # rank of last timestep.
        self.__ranks_last_step__ = -1

        self.t: int = 0
        self.M: int = 0
        return

    def reset(self):
        self.__ss_idx__ = 0
        self.__sketch_np__ = None
        self.__label_gt__ = -1
        self.__points_num__ = -1
        self.__ranks_last_step__ = -1
        self.t = 0
        self.M: int = 0
        return

    def set_cur_sketch(self, sketch_data, label_gt):
        """
        :param sketch_data: un-0-padded sketch_np
        :param label_gt: ground truth label
        :return:
        """
        assert self.__sketch_np__ is None and self.__label_gt__ == -1, \
            "SAEnv - set_cur_sketches: cur_sketches is not empty. Call reset() prior to set_cur_sketches()"

        self.__sketch_np__ = copy.deepcopy(sketch_data)
        self.__points_num__ = len(self.__sketch_np__)

        sketch_np_0padded = np.zeros(shape=(dataset.find_max_points_num(), 3))
        points_num = len(self.__sketch_np__)
        sketch_np_0padded[0: points_num] = self.__sketch_np__
        self.__sketch_np__ = sketch_np_0padded

        self.__label_gt__ = label_gt

        self.__ranks_last_step__ = -1
        self.t = 1
        self.M = max(self.__points_num__ // ss_len(), 1)
        return copy.deepcopy(self.__sketch_np__)

    def step(self, action):
        assert self.__sketch_np__ is not None and self.__label_gt__ != -1
        assert action in self.action_space
        assert 0 < self.t <= self.M

        start: int = self.__ss_idx__ * ss_len()
        end: int = self.__points_num__ \
            if (self.__ss_idx__ + 1) * ss_len() >= self.__points_num__ \
            else (self.__ss_idx__ + 1) * ss_len()

        assert end - start == ss_len() or (end - start < ss_len() and end == self.__points_num__)

        if action == 0:
            delta_x, delta_y = 0.0, 0.0

            for j in range(start, end):
                point = self.__sketch_np__[j]
                delta_x += point[0]
                delta_y += point[1]

            # remove 0-padded part
            rm_ss_sketch_np = np.zeros(shape=(len(self.__sketch_np__) - (end - start), 3))

            if start > 0:
                rm_ss_sketch_np[0:start][:] = self.__sketch_np__[0:start][:]
                rm_ss_sketch_np[start - 1][2] = 1.0

            rm_ss_sketch_np[start:][:] = self.__sketch_np__[end:][:]
            rm_ss_sketch_np[start][0] += delta_x
            rm_ss_sketch_np[start][1] += delta_y

            self.__sketch_np__ = rm_ss_sketch_np
        else:
            self.__ss_idx__ += 1

        if action == 0:
            self.__points_num__ = self.__points_num__ - (end - start)

        # issue: reward computation too slow
        reward, info = self.__reward__(action)
        self.t += 1
        return copy.deepcopy(self.__sketch_np__), reward, self.t - 1 >= self.M, info


    def __reward__(self, action):
        DIV_F: float = 1.0

        if action not in self.action_space:
            raise "SAEnv - step: argument `actions` contains at least one invalid action"

        reward: float = 0.
        w_r = (self.t - 1) / float(self.M) * rl_config.weight_ranked_final()
        w_b = 1.0 - w_r

        # Note: follow paper 3.1.4
        if self.t > self.M:
            return 0., None

        elif self.t == self.M:
            pred_prob = self.__classifier__(np.array([self.__sketch_np__]))
            pred_prob = pred_prob[0]
            pred_class = np.argmax(pred_prob)
            reward = 20.0 / DIV_F if pred_class == self.__label_gt__ else -20.0 / DIV_F  # todo: decrease recognized reward?
            reward = reward * w_b
            # print("w_r: {}, w_b: {}, final reward: {}".format(w_r, w_b, reward))

            return reward, None
        else:
            # Basic reward
            b_r: float = 1.0 / DIV_F if action == 0 else -5.0 / DIV_F

            # Ranked reward computation
            label_gt = self.__label_gt__
            pred_prob = self.__classifier__(np.array([self.__sketch_np__]))
            pred_prob = pred_prob[0]
            pred_class = np.argmax(pred_prob)
            pred_ranks = np.argsort(pred_prob)
            sorted_index = -1

            for k in range(len(pred_prob)):
                if pred_ranks[k] == label_gt:
                    sorted_index = k
                    break

            assert sorted_index != -1

            K: int = 9 #len(dataset.categories)

            # rank of current and last time step
            # i.e. C_t and C_t-1 in the paper
            rank_cur = K - (sorted_index + 1)
            rank_last = self.__ranks_last_step__

            # compute ranked reward coefficient
            c_t = rank_cur / K
            v_t = 1.0 - (K - (rank_cur - rank_last)) / (2.0 * K)
            rank_coeff = rl_config.w_c() * c_t + rl_config.w_v() * v_t

            # ranked reward and reward for the current timestep
            r_r = rank_coeff * b_r
            reward = w_b * b_r + w_r * r_r

            self.__ranks_last_step__ = rank_cur

            return reward, {"w_b": w_b, "w_r": w_r, "b_r": b_r, "r_r": r_r}
