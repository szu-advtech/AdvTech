from __future__ import absolute_import, division, print_function

import csv

import matplotlib.pyplot as plt

import tensorflow as tf
import configs
from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment

from rm_DRL import ClusterEnv
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network, actor_distribution_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()


def collect_episode(environment, policy, num_episodes, replay_buffer):
    episode_counter = 0
    environment.reset()

    while episode_counter < num_episodes:
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        traj = trajectory.from_transition(time_step, action_step, next_time_step)

        replay_buffer.add_batch(traj)

        if traj.is_boundary():
            episode_counter += 1


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def train_reinforce(

        num_iterations=20000,
        collect_episodes_per_iteration=2,
        replay_buffer_max_length=10000,
        fc_layer_params=(100,),
        learning_rate=1e-3,
        log_interval=200,
        num_eval_episodes=10,
        eval_interval=1000
):
    file = open(configs.root+'/output/avg_returns_'+configs.algo+'_beta_'+str(configs.beta)+'.csv', 'w', newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return"])

    train_py_env = ClusterEnv()
    eval_py_env = train_py_env

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v1.Variable(0)

    actor_net = actor_distribution_network.ActorDistributionNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    agent = reinforce_agent.ReinforceAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        actor_network=actor_net,
        optimizer=optimizer,
        normalize_returns=True,
        train_step_counter=train_step_counter)

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    agent.train = common.function(agent.train)

    agent.train_step_counter.assign(0)

    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        collect_episode(
            train_env, agent.collect_policy, collect_episodes_per_iteration, replay_buffer)

        experience = replay_buffer.gather_all()
        train_loss = agent.train(experience)
        replay_buffer.clear()

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss.loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            avg_return_writer.writerow([step, avg_return])
            returns.append(avg_return)




# def compute_avg_return(environment, policy, num_episodes=10):
#     total_return = 0.0
#     for _ in range(num_episodes):
#
#         time_step = environment.reset()
#         episode_return = 0.0
#
#         print('\n\n evaluation started \n')
#         while not time_step.is_last():
#             action_step = policy.action(time_step)
#             print('action: ', action_step.action)
#             time_step = environment.step(action_step.action)
#             episode_return += time_step.reward
#         total_return += episode_return
#         print('episode return: ', episode_return)
#
#     avg_return = total_return / num_episodes
#     return avg_return.numpy()[0]
#
#
# num_iterations = 40000  # @param
#
# initial_collect_steps = 1000  # @param
# collect_steps_per_iteration = 1  # @param
# replay_buffer_capacity = 100000  # @param
#
# fc_layer_params = (100,)
#
# batch_size = 128  # @param
# learning_rate = 1e-5  # @param
# log_interval = 200  # @param
#
# num_eval_episodes = 2  # @param
# eval_interval = 1000  # @param
#
# train_py_env = wrappers.TimeLimit(ClusterEnv(), duration=100)
# eval_py_env = wrappers.TimeLimit(ClusterEnv(), duration=100)
# # train_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)
# # eval_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)
#
# train_env = tf_py_environment.TFPyEnvironment(train_py_env)
# eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
#
# q_net = q_network.QNetwork(
#     train_env.observation_spec(),
#     train_env.action_spec(),
#     fc_layer_params=fc_layer_params)
#
# optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
#
# train_step_counter = tf.compat.v2.Variable(0)
#
# tf_agent = dqn_agent.DqnAgent(
#     train_env.time_step_spec(),
#     train_env.action_spec(),
#     q_network=q_net,
#     optimizer=optimizer,
#     td_errors_loss_fn=common.element_wise_squared_loss,
#     train_step_counter=train_step_counter)
#
# tf_agent.initialize()
#
# eval_policy = tf_agent.policy
# collect_policy = tf_agent.collect_policy
#
# replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#     data_spec=tf_agent.collect_data_spec,
#     batch_size=train_env.batch_size,
#     max_length=replay_buffer_capacity)
#
# replay_observer = [replay_buffer.add_batch]
#
# train_metrics = [
#     tf_metrics.NumberOfEpisodes(),
#     tf_metrics.EnvironmentSteps(),
#     tf_metrics.AverageReturnMetric(),
#     tf_metrics.AverageEpisodeLengthMetric(),
# ]
#
#
# def collect_step(environment, policy):
#     time_step = environment.current_time_step()
#     action_step = policy.action(time_step)
#     next_time_step = environment.step(action_step.action)
#     traj = trajectory.from_transition(time_step, action_step, next_time_step)
#
#     # Add trajectory to the replay buffer
#     replay_buffer.add_batch(traj)
#
#
# for _ in range(1000):
#     collect_step(train_env, tf_agent.collect_policy)
#
# dataset = replay_buffer.as_dataset(
#     num_parallel_calls=3,
#     sample_batch_size=batch_size,
#     num_steps=2).prefetch(3)
#
# driver = dynamic_step_driver.DynamicStepDriver(
#     train_env,
#     collect_policy,
#     observers=replay_observer + train_metrics,
#     num_steps=1)
#
# iterator = iter(dataset)
#
# print(compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes))
#
# tf_agent.train = common.function(tf_agent.train)
# tf_agent.train_step_counter.assign(0)
#
# final_time_step, policy_state = driver.run()
#
# for i in range(1000):
#     final_time_step, _ = driver.run(final_time_step, policy_state)
#
# episode_len = []
# for i in range(num_iterations):
#     final_time_step, _ = driver.run(final_time_step, policy_state)
#     # for _ in range(1):
#     #    collect_step(train_env, tf_agent.collect_policy)
#
#     experience, _ = next(iterator)
#     train_loss = tf_agent.train(experience=experience)
#     step = tf_agent.train_step_counter.numpy()
#
#     if step % log_interval == 0:
#         print('step = {0}: loss = {1}'.format(step, train_loss.loss))
#         episode_len.append(train_metrics[3].result().numpy())
#         print('Average episode length: {}'.format(train_metrics[3].result().numpy()))
#
#     if step % eval_interval == 0:
#         avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
#         print('step = {0}: Average Return = {1}'.format(step, avg_return))
# plt.plot(episode_len)
# plt.show()
