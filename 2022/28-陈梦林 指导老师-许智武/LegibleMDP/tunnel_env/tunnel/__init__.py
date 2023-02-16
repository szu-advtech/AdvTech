# from .tunnel import TunnelEnv, BuilderTunnelEnv, TwoGoalsTunnelEnv
import gym
# from .eval import EnvTraining
# from .wrappers import SlicedEmbeddings, SlicedConvolution
# from tunnel.env import TunnelEnv, BuilderTunnelEnv, TwoGoalsTunnelEnv
# from .env.tunnel import TunnelEnv,BuilderTunnelEnv,TwoGoalsTunnelEnv
# from .env.wrappers import SlicedEmbeddings,SlicedConvolution
# from .env.eval import EnvTraining
from gym.envs.registration import register
from tunnel.env import BuilderTunnelEnv
from tunnel.env import TwoGoalsTunnelEnv


def rect_shape(grid, x, y):
    return grid[x:x + 4, y:y + 16, :]

def rect_shape_l(grid, x, y):
    return grid[x:x + 2, y:y + 20, :]

def rect_shape_h(grid, x, y):
    return grid[x:x + 12, y:y + 2, :]


def square_shape(grid, x, y):
    return grid[x:x + 4, y:y + 4, :]

def square_shape_l(grid, x, y):
    return grid[x:x + 1, y:y + 20, :]






register(
    id='Tunnel-v0',
    entry_point='tunnel.env:BuilderTunnelEnv',
    max_episode_steps=2000,
    kwargs={'tunnel_length' : 170, 'tunnel_width' : 12, 'sight_distance' : 20, 'num_colors' : 4,
            'color_density' : 26, 'color_shapes' : [rect_shape], 'reward_color' : None,
            'obstacle_density' : 8, 'obstacle_shapes' : [square_shape, square_shape_l],
            'color_reward_amount' : 1, 'obstacle_reward' : -10, 'steer_reward' : -0.1, 'reset_on_obstacle_hit' : False,
            },
)

register(
    id='TestTunnel-v0',
    entry_point='tunnel.env:TwoGoalsTunnelEnv',
    max_episode_steps=2000,
    kwargs={'tunnel_length' : 50, 'tunnel_width' : 12, 'sight_distance' : 20, 'reset_on_obstacle_hit' : True,
            },
)


