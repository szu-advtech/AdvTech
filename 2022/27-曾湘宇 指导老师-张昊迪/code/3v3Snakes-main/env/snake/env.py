from core.env import Env
from snake_wrapper import SnakeEnvWrapper
ACTION_DIM = 3
STATE_SHAPE = 26

class SnakeEnv(Env):
    def __init__(self, gym_env, *args, **kwargs):
        super(SnakeEnv, self).__init__(*args, **kwargs)
        self.env_wrapper = SnakeEnvWrapper()
        self.action_dim = ACTION_DIM
        self.state_shape = STATE_SHAPE

    def reset(self):
        return self.env_wrapper.reset()

    def step(self, action, *args, **kwargs):
        return self.env_wrapper.step(action)

    def get_action_space(self):
        return self.action_dim

    def get_observation_space(self):
        return self.state_shape

    def calc_reward(self, *args, **kwargs):
        raise NotImplemented

    def render(self) -> None:
        self.env_wrapper.render()

    def close(self) -> None:
        self.env_wrapper.close()
