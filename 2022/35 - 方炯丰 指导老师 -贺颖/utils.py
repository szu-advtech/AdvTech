from collections import deque
from typing import Iterable
import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F


class EpisodeStore:
    def __init__(self, maxlen=int(450)):
        self.episodes = deque(maxlen=maxlen)
        self.n_episodes = 0
        self.maxlen = maxlen

    def add(self, obs, act, rew, free_energy, done):
        self.episodes.append(Episode(obs, act, rew, free_energy, done))
        self.n_episodes = min(self.n_episodes + 1, self.maxlen)

    def sample_paths(self, n_paths, path_length, device='cuda' if torch.cuda.is_available() else 'cpu', balance_ends=True):
        #选取n_path条路径
        episode_indexes = np.random.randint(0, self.n_episodes, n_paths) #low high size
        #从每个路径中选取，step步的数据
        if balance_ends:
            #从所有episode选取，对超长的进行判断
            path_indexes = [np.random.randint(0, len(self.episodes[ei]) + 1) for ei in episode_indexes]
            path_indexes = [pi if pi < len(self.episodes[ei]) - path_length + 1 else len(self.episodes[ei]) - path_length for pi, ei in zip(path_indexes, episode_indexes)]
        #防止尾部数据过长，进行截取调整
        else:
            #限制抽取长度
            path_indexes = [np.random.randint(0, len(self.episodes[ei]) - path_length + 1) for ei in episode_indexes]
        #截取随机坐标的连续空间属性
        path_obs = [self.episodes[ei].obs[pi:pi+path_length + 1] for ei, pi in zip(episode_indexes, path_indexes)]
        path_act = [self.episodes[ei].act[pi:pi+path_length] for ei, pi in zip(episode_indexes, path_indexes)]
        path_rew = [self.episodes[ei].rew[pi:pi+path_length] for ei, pi in zip(episode_indexes, path_indexes)]
        path_done = [self.episodes[ei].done[pi:pi+path_length] for ei, pi in zip(episode_indexes, path_indexes)]
        #返回torch
        return torch.Tensor(np.stack(path_obs, axis=1)).to(device), torch.Tensor(np.stack(path_act, axis=1)).to(device), torch.Tensor(np.stack(path_rew, axis=1)).to(device), torch.Tensor(np.stack(path_done, axis=1)).to(device)

class Episode:
    def __init__(self, obs, act, rew, free_energy, done):
        self._observations = np.array(obs)
        self._actions = np.array(act)
        self._rewards = np.array(rew)
        self._free_energy = np.array(free_energy)
        self._done = np.array(done)

    def __len__(self):
        return len(self._actions)

    @property
    def obs(self):
        return self._observations

    @property
    def act(self):
        return self._actions
    
    @property
    def rew(self):
        return self._rewards

    @property
    def done(self):
        return self._done

class RandomPolicy:
    def __init__(self, action_space, policy_lookahead=1, device='cuda' if torch.cuda.is_available() else 'cpu'):
        assert policy_lookahead > 0
        self._policy_lookahead = policy_lookahead
        self._action_space = action_space
        self._policy_actions = [torch.from_numpy(action_space.sample()).to(device) for _ in range(policy_lookahead)]
        self.device = device
        self._counter = 0

    def __call__(self, state):
        self._counter = self._counter + 1 % self._policy_lookahead
        return torch.from_numpy(self._action_space.sample()).to(self.device)
#真实自由能，预测自由能，gama
def lambda_dreamer_values(rewards, value_preds, gamma = 0.99, gae_lamda = 0.95):  #reward是每个状态的自由能
    lambda_returns = torch.zeros_like(rewards)
    if type(gamma) in [int, float]:
        gamma = torch.ones_like(rewards) * gamma
    lambda_returns[-1] = rewards[-1] + gamma[-1] * value_preds[-1] #最后一步的未来效用
    for step in reversed(range(rewards[:-1].size(0))):#倒序更新每一个时间步的折扣因子，采用gae的方式进行更新，减少预测误差
        lambda_returns[step] = rewards[step] + gamma[step] * ( (1 - gae_lamda) * value_preds[step + 1] + gae_lamda * lambda_returns[step + 1]) #（无偏加有偏）×gamma +reward
    return lambda_returns #返回每个状态的未来自由能

def get_parameters(modules: Iterable[Module]):
    model_parameters = []
    for module in modules:
        model_parameters += list(module.parameters())
    return model_parameters

class FreezeParameters:
  def __init__(self, modules: Iterable[Module]):
      self.modules = modules
      self.param_states = [p.requires_grad for p in get_parameters(self.modules)]

  def __enter__(self):
      for param in get_parameters(self.modules):
          param.requires_grad = False

  def __exit__(self, exc_type, exc_val, exc_tb):
      for i, param in enumerate(get_parameters(self.modules)):
          param.requires_grad = self.param_states[i]

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count

from functools import wraps
import torch.nn as nn
import torch
class _SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# The SiLu activation (also known as Swish) was introduced in Pytorch 1.8. If not available, provide own implementation
if hasattr(torch.nn, 'SiLU'):
    SiLU = torch.nn.SiLU
else:
    SiLU = _SiLU

def retry(func):
    """
    A Decorator to retry a function for a certain amount of attempts
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        attempts = 0
        max_attempts = 100
        while attempts < max_attempts:
            try:
                return func(*args, **kwargs)
            except (OSError, PermissionError):
                attempts += 1
        raise OSError("Retry failed")

    return wrapper