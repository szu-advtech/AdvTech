# It's a wrapper around an environment and an agent that allows you to train the agent in the environment
# It's a wrapper around an environment and an agent that allows you to train the agent in the environment
#它是一个围绕环境的包装器和一个允许您在环境中训练代理的代理
from collections import deque, defaultdict
from tensorboardX import SummaryWriter
import numpy as np
import torch
import shutil
import os
import time



class EnvTraining:



    def __init__(self, env, agent, logs_directory='./logs', render=True, max_num_steps=10000):
        self.env = env
        self.agent = agent

        seed = str(time.time())
        self.logs_directory = f'{logs_directory}/{seed}'
        os.mkdir(self.logs_directory)
        self.writer = SummaryWriter(str(self.logs_directory))
        self.agent.set_writer(self.writer)
        # It's setting the maximum number of steps in an episode.
        self.max_num_steps = int(max_num_steps)
        self._render = render
        self.logs = {}
        self.global_logs = defaultdict(lambda : deque(maxlen=200))

    def log_avg(self, key):
        return np.mean(self.global_logs[key])

    def render(self):
        if self._render:
            self.env.render()


    def train(self, num_episodes, callbacks=None):
        """
        The function takes in the number of episodes to train for, and a list of callbacks.

        For each episode, the function resets the environment, and then for each step in the episode, the agent acts,
        records the state, action, reward, next state, and whether the episode is done, and then renders the environment.

        If the episode is done, the function logs the accumulated reward, max accumulated reward, epsilon, and number of
        steps.

        Then, for each callback in the list of callbacks, the function calls the callback with the episode number and the
        logs.

        Finally, the function calls the on_episode_end function.

        Let's take a look at the on_episode_end function.

        :param num_episodes: The number of episodes to train for
        :param callbacks: A list of callbacks that will be called at the end of each episode
        """
        """
        该函数接收要训练的集数和回调列表。
        对于每一集，该功能都会重置环境，然后对于每一集中的每一步，代理都会采取行动，
        记录状态、动作、奖励、下一个状态以及该集是否完成，然后渲染环境。
        如果情节完成，该函数将记录累计奖励、最大累计奖励、epsilon和步骤。
        然后，对于回调列表中的每个回调，该函数调用带有插曲号和
        日志。
        最后，该函数调用on_episode_end函数。
        让我们看看on_episode_end函数。
        ：param num_septs：要训练的集数
        ：param callbacks：将在每集结束时调用的回调列表
        """
        env = self.env
        agent = self.agent
        logs = self.global_logs

        if callbacks is None:
            callbacks = []

        for e in range(int(num_episodes)):
            state = env.reset()
            state_tmp = np.concatenate((state['colors'], np.expand_dims(state['obstacles'], axis=2)), axis=2)
            state_tmp = np.concatenate((state_tmp, np.expand_dims(state['position'], axis=2)), axis=2)
            # state_tmp=torch.tensor(state_tmp)
            state=state_tmp
            # print(f"state:{state}")
            # print(f"state_tmp:{state_tmp}")
            self.render()
            for step in range(self.max_num_steps):
                # It's getting the action from the agent.
                # 它从代理获取动作。

                action = agent.act(state)
                next_state, reward, done, info = env.step(action)

                agent.record(state, action, reward, next_state, done, info)
                self.render()

                if done:
                    logs['accumulated_reward'] += info['accumulated_reward'],
                    logs['max_accumulated_reward'] += info['max_accumulated_reward'],
                    logs['epsilon'] += agent.exploration_policy.epsilon,
                    logs['steps'] += step + 1,
                    print(f'Episode: {e}  Reward: {int(self.log_avg("accumulated_reward"))}/{int(self.log_avg("max_accumulated_reward"))}')
                    break
                print(state)
                state = next_state

            for cbk in callbacks:
                cbk(e, logs)

            self.on_episode_end(e)


    def on_episode_end(self, episode):
        self.print_logs(episode)
        self.agent.save('./models/agent')

    def print_logs(self, episode, prefix=None):
        logs = self.global_logs
        for k in logs.keys():
            v = self.log_avg(k)
            if prefix is not None:
                k = f'{prefix}/{k}'
            self.writer.add_scalar(k, v, episode)
        self.writer.flush()