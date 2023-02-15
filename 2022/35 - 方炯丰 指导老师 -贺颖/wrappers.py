import gym
import numpy as np
import torch
import threading

from PIL import Image

def make_env(suite='dmc', task_name='cartpole_balance', **kwargs):
    if suite == 'dmc': #创建reacher环境，连续动作，附加动作repeat
      action_repeat = 2
      timelimit = 1000
  
      if task_name == 'reacher_easy_13':
        env = DeepMindControlFixed(task_name, action_repeat = action_repeat, **kwargs)
      else:
        env = DeepMindControl(task_name, action_repeat = action_repeat, **kwargs)
  
      env = NormalizeActions(env)
      env = TimeLimit(env, timelimit)
    elif suite == 'minigrid_pixels':  #创建minigrid环境，one-hot动作
      env = Minigrid(task_name, size=kwargs['grid_size'], pixels=True)
      env = OneHotAction(env)
    else:
      raise NotImplementedError(suite)
    return env

def get_scaled_obs(timestep, device, is_minigrid=False,):
  if is_minigrid:  #三通道图片
    obs = torch.nn.functional.pad(torch.from_numpy(timestep['image'] / 255 - 0.5 ).float().permute(2,0,1).to(device), (0, 1, 0, 1))
  else:
    obs = torch.from_numpy(timestep['image'] / 255 - 0.5).float().to(device) 
  return obs

class DeepMindControl:

  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, **kwargs):
    if name.startswith("point_mass"):
      domain = "point_mass"
      task = name.split('_')[-1]
    else:
      domain, task = name.split('_', 1)
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    if isinstance(domain, str):
      from dm_control import suite
      self._env = suite.load(domain, task)
    else:
      assert task is None
      self._env = domain()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  @property
  def observation_space(self):
    spaces = {}
    for key, value in self._env.observation_spec().items():
      spaces[key] = gym.spaces.Box(
          -np.inf, np.inf, value.shape, dtype=np.float32)
    spaces['image'] = gym.spaces.Box(
        0, 255, self._size + (3,), dtype=np.uint8)
    return gym.spaces.Dict(spaces)

  @property
  def action_space(self):
    spec = self._env.action_spec()
    return gym.spaces.Box(spec.minimum, spec.maximum, dtype=np.float32)

  def step(self, action):
    assert np.isfinite(action).all(), action
    reward = 0
    for _ in range(self._action_repeat):
      time_step = self._env.step(action)
      reward += time_step.reward or 0
      if time_step.last():
        break
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    done = time_step.last()
    info = {'discount': np.array(time_step.discount, np.float32)}
    return obs, reward, done, info

  def reset(self):
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def render(self, *args, **kwargs):
    if kwargs.get('mode', 'rgb_array') != 'rgb_array':
      raise ValueError("Only render mode 'rgb_array' is supported.")
    return self._env.physics.render(*self._size, camera_id=self._camera).transpose(2, 0, 1).copy()

class DeepMindControlFixed(DeepMindControl):
  def __init__(self, name, action_repeat=1, size=(64, 64), camera=None, visualize_reward=False, **kwargs):
    domain, task = name.split('_', 1)
    self._name = name
    self._visualize_reward = visualize_reward
    if domain == 'cup':  # Only domain with multiple words.
      domain = 'ball_in_cup'
    self.domain = domain
    self.load_env()
    self._action_repeat = action_repeat
    self._size = size
    if camera is None:
      camera = dict(quadruped=2).get(domain, 0)
    self._camera = camera

  def reset(self):
    self._env.close()
    from dm_control import suite
    self.load_env()
    time_step = self._env.reset()
    obs = dict(time_step.observation)
    obs['image'] = self.render()
    return obs

  def load_env(self):
    if self._name == 'reacher_easy_13':
      from dm_control import suite
      self._env = suite.load(domain_name="reacher", task_name="easy", task_kwargs={'random':13}, visualize_reward=self._visualize_reward)
    else:
      raise NotImplementedError()

class NormalizeActions:

  def __init__(self, env): #动作归一化
    self._env = env
    self._mask = np.logical_and( #逻辑与
        np.isfinite(env.action_space.low), #判断数据是否为无限，有限为true，否则为false需要归一化
        np.isfinite(env.action_space.high))
    self._low = np.where(self._mask, env.action_space.low, -1)  #判断， bool x y    true=x  false=y
    self._high = np.where(self._mask, env.action_space.high, 1)

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    return gym.spaces.Box(low, high, dtype=np.float32)

  def step(self, action):
    original = (action + 1) / 2 * (self._high - self._low) + self._low
    original = np.where(self._mask, original, action)
    return self._env.step(original)

class TimeLimit:

  def __init__(self, env, duration):
    self._env = env
    self._duration = duration
    self._step = None

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    assert self._step is not None, 'Must reset environment.'
    obs, reward, done, info = self._env.step(action)
    self._step += 1
    if self._step >= self._duration:
      done = True
      if 'discount' not in info:
        info['discount'] = np.array(1.0).astype(np.float32)
      self._step = None
    return obs, reward, done, info

  def reset(self):
    self._step = 0
    return self._env.reset()

class SelectAction:

  def __init__(self, env, key):
    self._env = env
    self._key = key

  def __getattr__(self, name):
    return getattr(self._env, name)

  def step(self, action):
    return self._env.step(action[self._key])

class OneHotAction:

  def __init__(self, env):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    self._env = env

  def __getattr__(self, name):
    return getattr(self._env, name)

  @property
  def action_space(self):
    shape = (self._env.action_space.n,)
    space = gym.spaces.Box(low=0, high=1, shape=shape, dtype=np.float32)
    space.sample = self._sample_action
    return space

  def step(self, action):
    index = np.argmax(action).astype(int)
    reference = np.zeros_like(action)
    reference[index] = 1
    if not np.allclose(reference, action):
      raise ValueError(f'Invalid one-hot action:\n{action}')
    return self._env.step(index)

  def reset(self):
    return self._env.reset()

  def _sample_action(self):
    actions = self._env.action_space.n
    index = self._random.randint(0, actions)
    reference = np.zeros(actions, dtype=np.float32)
    reference[index] = 1.0
    return reference

class Minigrid:

  LOCK = threading.Lock()

  def __init__(
      self, name, random=False, size=None, pixels=False):
    import gym
    import gym_minigrid

    self._task_name = name
    task_name = ''.join(word.title() for word in name.split('_'))

    env_name = f'MiniGrid-{task_name}'
    if random:
      env_name += '-Random'
    if size is not None:
      env_name += f'-{size}'
    env_name += '-v0'

    with self.LOCK:
        self._env = gym.make(env_name)
        if pixels:
          self._env = gym_minigrid.wrappers.RGBImgPartialObsWrapper(self._env, tile_size=9) # Get pixel observations

    self._action_repeat = 1
    self._size = size
    shape = self._env.observation_space.shape
    self._shape = shape
    self._random = np.random.RandomState(seed=None)
    self._env_name = env_name
    if self._task_name == 'empty':
      self.goal_poses = { 0: torch.load(f'preferred_states/empty_goal_dir_0.pt')*255,
                          1: torch.load(f'preferred_states/empty_goal_dir_1.pt')*255}

  @property
  def observation_space(self):
    return self._env.observation_space

  @property
  def action_space(self):
    return self._env.action_space

  def close(self):
    return self._env.close()

  def reset(self):
    with self.LOCK:
      return self._env.reset()

  def step(self, action):
    timestep, rew, done, info = self._env.step(action)
    # If we don't do this, the goal green square is not visible under the red arrow
    if self._task_name == 'empty' and rew > 0.:
        timestep['image'] = self.goal_poses[self._env.unwrapped.agent_dir]

    return (timestep, rew, done, info) 

  def render(self, mode='human'):
    return self._env.render(mode)