from balloon_learning_environment.env import balloon_env  # Registers the environment.
from balloon_learning_environment.eval import eval_lib
from balloon_learning_environment.utils import run_helpers
from balloon_learning_environment.eval import suites
from balloon_learning_environment import train_lib
import gym

agent_name = 'finetune_perciatelli'
env = gym.make('BalloonLearningEnvironment-v0')
run_helpers.bind_gin_variables(agent_name)
agent = run_helpers.create_agent(agent_name,
                                env.action_space.n,
                                env.observation_space.shape)

train_lib.run_training_loop(
        '/root/anaconda3/envs/blenv/lib/python3.8/site-packages/balloon_learning_environment/blesh/DL100E',  # The experiment root path.
        env,
        agent,
        num_iterations=25,
        max_episode_length=960,  # 960 steps is 2 days, the default amount.
        collector_constructors=[])  # Specify some collectors to log training stats.         

eval_results = eval_lib.eval_agent(
        agent,
        env,
        eval_suite=suites.get_eval_suite('small_eval'))

import time
date = time.time()
import os
file = open(str(date),'w')
file.write(str(eval_results))
print(eval_results)