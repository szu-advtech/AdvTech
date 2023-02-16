# It takes a tunnel observation and transforms it into a 3D array
import gym
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt, matplotlib
from matplotlib.animation import FuncAnimation

from tunnel.env.tunnel import TunnelEnv




class TunnelWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        assert isinstance(env.unwrapped, TunnelEnv)
        super().__init__(env)


    def observation(self, observation):
        raise NotImplementedError

    def check_obstacle(self, x, y):
        return self.unwrapped.check_obstacles(x, y)

    def get_reward(self, color):
        return self.unwrapped.get_reward(color)

    def get_max_reward(self, color):
        return self.unwrapped.get_max_reward(color)

    @property
    def reward_color(self):
        return self.unwrapped.reward_color

    @property
    def n_colors(self):
        return self.unwrapped.n_colors


class ExtensionTunnelWrapper(TunnelWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.wrapper_reward_color = None
        self.observation_space = self.get_observation_space()
        self.noise = 0

    def observation(self, observation):
        reward_color = observation['reward_color']
        if self.wrapper_reward_color is not None:
            reward_color = self.wrapper_reward_color
        obs = self.get_color_observation(observation, reward_color)

        if self.noise > 0:
            h, w = self.unwrapped.size
            w = self.unwrapped.sight_distance
            mask = (np.random.uniform(0,1, size=(h,w)) > self.noise).astype('float').reshape((h,w,1))
            obs[:,:,:-1] *=  mask
        return obs


    def get_color_observation(self, observation, color):
        """
        Extracts the slice of the selected color (HxS) and transforms it into a cube (HxSxS)
        H: height, S: sight distance
        :param observation:
        :param color:
        :return:
        H x S x (2*S+1)    (color + obstacles + position)
                         (H x S x S)  (H x S x S)  (H x S x S)
        """
        pos = observation['position']
        tunnel = observation['colors'][:, :, color]
        obstacles = observation['obstacles']

        t_color = self.get_extended(tunnel, 0)
        t_obstacles = self.get_extended(obstacles, 1)
        t_pos = self.get_extended(pos, 2)
        obs = self.aggregate(t_color, t_obstacles, t_pos)
        return obs

    def aggregate(self, t_color, t_obstacles, t_pos):
        return np.concatenate([t_color, t_obstacles, t_pos], axis=-1)

    def get_extended(self, mat, cell_type):
        raise NotImplementedError

    def get_observation_space(self):
        raise NotImplementedError

class SlicedConvolution(ExtensionTunnelWrapper):

    def __init__(self, env):
        super().__init__(env)
        sshape = np.array(self.state_shape)
        sshape[-1] = (sshape[-1] - 1) /2 + 1

    def get_observation_space(self):
        return Box(0, 1, shape=(self.env.tunnel_width,
                                self.env.sight_distance,
                                3*self.env.sight_distance)
                   )

    def get_extended(self, mat, cell_type):
        d = mat.shape[-1]
        obs = np.zeros(mat.shape + (d,))
        for i in range(d):
            obs[:, i, i] = mat[:, i]
        return obs




# It takes the observation from the environment, and returns a new observation that is a slice of the original observation
#它从环境中获取观察结果，并返回一个新的观察结果，该观察结果是原始观察结果的一部分
class SlicedEmbeddings(ExtensionTunnelWrapper):

    def __init__(self, env):
        assert isinstance(env.unwrapped, TunnelEnv)
        super().__init__(env)

    def get_observation_space(self):
        return Box(0, 1, shape=(self.env.tunnel_width,
                                                  self.env.sight_distance,
                                                  3)
                                     )



    def aggregate(self, t_color, t_obstacles, t_pos):
        t_pos[:, 1:, :] = 0 # Position appears only in left-most column
        return super().aggregate(t_color, t_obstacles, t_pos)

    def get_extended(self, mat, cell_type):
        mat = np.copy(mat)
        mat[mat == 1] = 1
        return np.expand_dims(mat, -1)

