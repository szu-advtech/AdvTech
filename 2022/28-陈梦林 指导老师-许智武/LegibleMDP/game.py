import gym
import tunnel
import time

env=gym.make('Tunnel-v0')
# env=gym.make('TestTunnel-v0')
env.reset()
env.render()
time.sleep(10)
env.close()
# envs = [env.id for env in gym.envs.registry.all()]
# for env in envs:
#     print(env)