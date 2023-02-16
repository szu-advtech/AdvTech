from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import csv
import logging
import abc
import tensorflow as tf
import numpy as np
import cluster
import configs
from queue import PriorityQueue
import defines as defs
from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()

# logging.basicConfig(level=logging.DEBUG, filename=configs.root+'/output/'+configs.algo+'.log', filemode='w')
logging.basicConfig(level=logging.INFO, filename=configs.root+'/output/'+configs.algo+'.log', filemode='w')

episodes = 1


class ClusterEnv(py_environment.PyEnvironment):

    def __init__(self):
        self.file_result = open(configs.root + '/output/results_' + configs.algo + '.csv', 'a+', newline='')
        self.episode_reward_writer = csv.writer(self.file_result, delimiter=',')
        self.episode_reward_writer.writerow(["Episode", "Reward", "Cost", "AVGtime", "GoodPlacement"])
        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=9, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(cluster.features,), dtype=np.int32, minimum=cluster.cluster_state_min,
            maximum=cluster.cluster_state_max,
            name='observation')
        self._state = copy.deepcopy(cluster.cluster_state_init)
        self._episode_ended = False
        self.reward = 0
        self.vms = copy.deepcopy(cluster.VMS)
        self.jobs = copy.deepcopy(cluster.JOBS)
        self.clock = self.jobs[0].arrival_time
        self.job_idx = 0
        self.job_queue = PriorityQueue()
        self.episode_success = False
        self.good_placement = 0

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._state = copy.deepcopy(cluster.cluster_state_init)
        self._episode_ended = False
        self.reward = 0
        self.vms = copy.deepcopy(cluster.VMS)
        self.jobs = copy.deepcopy(cluster.JOBS)
        self.clock = self.jobs[0].arrival_time
        self.job_idx = 0
        self.job_queue = PriorityQueue()
        self.episode_success = False
        self.good_placement = 0
        return ts.restart(np.array(self._state, dtype=np.int32))

    def _step(self, action):

        global episodes
        if self._episode_ended:
            return self.reset()

        if action > 9 or action < 0:
            raise ValueError('`action` should be in 0 to 9.')

        elif action == 0:
            if self.jobs[self.job_idx].ex_placed > 0:
                self.reward = (-50)
                self._episode_ended = True
                logging.info(
                    "CLOCK: {}: Partial Executor Placement for a Job. Episode Ended\n\n".format(self.clock))
            elif self.job_queue.empty():
                self.reward = (-200)
                self._episode_ended = True
                logging.info(
                    "CLOCK: {}: No Executor Placement When No Job was Running. Episode Ended\n\n".format(self.clock))
            else:
                self.reward = -1
                _, y = self.job_queue.get()
                self.clock = y.finish_time
                self.finish_one_job(y)

        else:
            logging.info("CLOCK: {}: Action: {}".format(self.clock, action))
            if self.execute_placement(action):
                if self.check_enough_cluster_resource():
                    self.reward = 1
                else:
                    self.reward = (-200)
                    self._episode_ended = True
                    logging.info(
                        "CLOCK: {}: Optimistic Executor Placement will lead to cluster resource shortage. Episode "
                        "Ended\n\n".format(self.clock))
            else:
                self.reward = (-200)
                self._episode_ended = True
                logging.info("CLOCK: {}: Invalid Executor Placement, Episode Ended\n\n".format(self.clock))

        if self._episode_ended:

            epi_cost = cluster.max_episode_cost
            epi_avg_job_duration = cluster.min_avg_job_duration + \
                                   cluster.min_avg_job_duration * float(configs.placement_penalty) / 100

            if self.episode_success:
                epi_cost = self.calculate_vm_cost()
                cost_normalized = 1 - (epi_cost / cluster.max_episode_cost)
                cost_reward = cost_normalized * configs.beta

                epi_avg_job_duration = self.calculate_avg_time()
                max_avg_job_duration = cluster.min_avg_job_duration + cluster.min_avg_job_duration * (configs.placement_penalty/100.0)
                time_normalized = 1 - (epi_avg_job_duration-cluster.min_avg_job_duration) / (max_avg_job_duration-cluster.min_avg_job_duration)
                time_reward = time_normalized * (1 - configs.beta)

                self.reward = configs.fixed_episodic_reward * (cost_reward + time_reward)
                logging.info("CLOCK: {}: ****** Episode ended Successfully! \n\n".format(self.clock))
                logging.info("cost normalized: {}, cost reward: {}, time normalized: {}, "
                              "time reward: {}, final reward: {}\n\n".format(cost_normalized, cost_reward,
                                                                             time_normalized, time_reward,
                                                                             self.reward))

            self.episode_reward_writer.writerow([episodes, self.reward, epi_cost, epi_avg_job_duration, self.good_placement])
            episodes += 1
            return ts.termination(np.array(self._state, dtype=np.int32), self.reward)

        else:
            return ts.transition(
                np.array(self._state, dtype=np.int32), reward=self.reward, discount=.9)

    def finish_one_job(self, finished_job):
        finished_job.finished = True
        finished_job.running = False
        for i in range(len(finished_job.ex_placement_list)):
            vm = self.vms[finished_job.ex_placement_list[i]]
            vm.cpu_now += finished_job.cpu
            vm.mem_now += finished_job.mem

        self._state = cluster.gen_cluster_state(self.job_idx, self.jobs, self.vms)
        logging.info("CLOCK: {}: Finished execution of job: {}".format(self.clock, finished_job.id))
        logging.debug("CLOCK: {}: Current Cluster State: {}".format(self.clock, self._state))

    def execute_placement(self, action):

        current_job = self.jobs[self.job_idx]
        vm_idx = action - 1

        if current_job.cpu > self.vms[vm_idx].cpu_now or current_job.mem > self.vms[vm_idx].mem_now:
            return False

        self.vms[vm_idx].cpu_now -= current_job.cpu
        self.vms[vm_idx].mem_now -= current_job.mem
        current_job.ex_placed += 1
        current_job.ex_placement_list.append(vm_idx)

        if current_job.ex_placed == current_job.ex:
            logging.info("CLOCK: {}: Finished placement of job: {}".format(self.clock, current_job.id))

            if configs.pp_apply == 'true':
                if current_job.type == 3:
                    if len(set(current_job.ex_placement_list)) != 1:
                        logging.debug("***** Bad placement for type 3 job. Executors: {}, Machines used: {}".format(
                            current_job.ex_placed, len(set(current_job.ex_placement_list))))
                        duration_increase = current_job.duration * float(configs.placement_penalty) / 100
                        current_job.duration += duration_increase
                    else:
                        self.good_placement += 1
                        logging.debug("***** Good placement for type 3 job. Executors: {}, Machines used: {}".format(
                            current_job.ex_placed, len(set(current_job.ex_placement_list))))

                else:
                    if len(set(current_job.ex_placement_list)) < current_job.ex_placed:
                        logging.debug("***** Bad placement for type 1 or 2 job. Executors: {}, Machines used: {}".format
                                      (current_job.ex_placed, len(set(current_job.ex_placement_list))))
                        duration_increase = current_job.duration * float(configs.placement_penalty) / 100
                        current_job.duration += duration_increase
                    else:
                        self.good_placement += 1
                        logging.debug(
                            "***** Good placement for type 1 or 2 job. Executors: {}, Machines used: {}".format(
                                current_job.ex_placed, len(set(current_job.ex_placement_list))))

            current_job.running = True
            current_job.start_time = self.clock
            current_job.finish_time = self.clock + current_job.duration

            for i in range(len(current_job.ex_placement_list)):
                if current_job.start_time > self.vms[current_job.ex_placement_list[i]].stop_use_clock:
                    self.vms[current_job.ex_placement_list[i]].used_time += current_job.duration
                    self.vms[current_job.ex_placement_list[i]].stop_use_clock = current_job.finish_time
                else:
                    if current_job.finish_time > self.vms[current_job.ex_placement_list[i]].stop_use_clock:
                        self.vms[current_job.ex_placement_list[i]].used_time += (
                                current_job.finish_time - self.vms[current_job.ex_placement_list[i]].stop_use_clock)
                        self.vms[current_job.ex_placement_list[i]].stop_use_clock = current_job.finish_time

            self.job_queue.put((current_job.finish_time, current_job))
            if self.job_idx + 1 == len(self.jobs):
                self._episode_ended = True
                self.episode_success = True
                return True
            self.job_idx += 1
            self.clock = self.jobs[self.job_idx].arrival_time

            while True:
                if self.job_queue.empty():
                    break
                _, next_finished_job = self.job_queue.get()
                if next_finished_job.finish_time <= self.clock:
                    self.finish_one_job(next_finished_job)
                else:
                    self.job_queue.put((next_finished_job.finish_time, next_finished_job))
                    break

        self._state = cluster.gen_cluster_state(self.job_idx, self.jobs,
                                                self.vms)
        logging.debug("CLOCK: {}: Current Cluster State: {}".format(self.clock, self._state))
        return True

    def check_enough_cluster_resource(self):
        current_job = self.jobs[self.job_idx]
        possible_placement = 0
        remaining_placement = current_job.ex - current_job.ex_placed
        for i in range(len(self.vms)):
            possible_placement += min(self.vms[i].cpu_now / current_job.cpu, self.vms[i].mem_now / current_job.mem)

        return possible_placement >= remaining_placement

    def check_episode_end(self):
        current_job = self.jobs[self.job_idx]
        if self.job_idx + 1 == len(self.jobs) and current_job.ex == current_job.ex_placed:
            self._episode_ended = True

    def calculate_vm_cost(self):
        cost = 0
        for i in range(len(self.vms)):
            cost += (self.vms[i].price * self.vms[i].used_time)
            logging.info("VM: {}, Price: {}, Time: {}".format(i, self.vms[i].price, self.vms[i].used_time))
        logging.info("***Episode VM Cost: {}".format(cost))
        return cost

    def calculate_avg_time(self):
        time = 0
        for i in range(len(self.jobs)):
            time += self.jobs[i].duration
            logging.debug("Job: {}, Duration: {}".format(self.jobs[i].id, self.jobs[i].duration))
        avg_time = float(time) / len(self.jobs)
        logging.info("***Episode AVG Job Duration: {}".format(avg_time))
        return avg_time




# class CardGameEnv(py_environment.PyEnvironment):
#
#     def __init__(self):
#         self._action_spec = array_spec.BoundedArraySpec(
#             shape=(), dtype=np.int32, minimum=0, maximum=1, name='action')
#         self._observation_spec = array_spec.BoundedArraySpec(
#             shape=(1,), dtype=np.int32, minimum=0, name='observation')
#         self._state = 0
#         self._episode_ended = False
#
#     def action_spec(self):
#         return self._action_spec
#
#     def observation_spec(self):
#         return self._observation_spec
#
#     def _reset(self):
#         self._state = 0
#         self._episode_ended = False
#         return ts.restart(np.array([self._state], dtype=np.int32))
#
#     def _step(self, action):
#
#         if self._episode_ended:
#             # The last action ended the episode. Ignore the current action and start
#             # a new episode.
#             return self.reset()
#
#         # Make sure episodes don't go on forever.
#         if action == 1:
#             self._episode_ended = True
#         elif action == 0:
#             new_card = np.random.randint(1, 11)
#             self._state += new_card
#         else:
#             raise ValueError('`action` should be 0 or 1.')
#
#         if self._episode_ended or self._state >= 21:
#             reward = self._state - 21 if self._state <= 21 else -21
#             return ts.termination(np.array([self._state], dtype=np.int32), reward)
#         else:
#             return ts.transition(
#                 np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
#
#
# environment = CardGameEnv()
# utils.validate_py_environment(environment, episodes=5)
#
# get_new_card_action = 0
# end_round_action = 1
#
# environment = CardGameEnv()
# time_step = environment.reset()
# print(time_step)
# cumulative_reward = time_step.reward
#
# for _ in range(3):
#     time_step = environment.step(get_new_card_action)
#     print(time_step)
#     cumulative_reward += time_step.reward
#
# time_step = environment.step(end_round_action)
# print(time_step)
# cumulative_reward += time_step.reward
# print('Final Reward = ', cumulative_reward)

# def compute_avg_return(environment, policy, num_episodes=10):
#     total_return = 0.0
#     for _ in range(num_episodes):
#
#         time_step = environment.reset()
#         episode_return = 0.0
#
#         while not time_step.is_last():
#             action_step = policy.action(time_step)
#             time_step = environment.step(action_step.action)
#             episode_return += time_step.reward
#             total_return += episode_return
#
#     avg_return = total_return / num_episodes
#     return avg_return.numpy()[0]
#
#
# num_iterations = 10000  # @param
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
# train_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)
# eval_py_env = wrappers.TimeLimit(GridWorldEnv(), duration=100)
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
# # for _ in range(1000):
# #        collect_step(train_env, tf_agent.collect_policy)
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
# class GridWorldEnv(py_environment.PyEnvironment):
#
#     def __init__(self):
#         self._action_spec = array_spec.BoundedArraySpec(
#             shape=(), dtype=np.int32, minimum=0, maximum=3, name='action')
#         self._observation_spec = array_spec.BoundedArraySpec(
#             shape=(4,), dtype=np.int32, minimum=[0,0,0,0],maximum=[5,5,5,5], name='observation')
#         self._state=[0,0,5,5] #represent the (row, col, frow, fcol) of the player and the finish
#         self._episode_ended = False
#
#     def action_spec(self):
#         return self._action_spec
#
#     def observation_spec(self):
#         return self._observation_spec
#
#     def _reset(self):
#         self._state=[0,0,5,5]
#         self._episode_ended = False
#         return ts.restart(np.array(self._state, dtype=np.int32))
#
#     def _step(self, action):
#
#         if self._episode_ended:
#             return self.reset()
#
#         self.move(action)
#
#         if self.game_over():
#             self._episode_ended = True
#
#         if self._episode_ended:
#             if self.game_over():
#                 reward = 100
#             else:
#                 reward = 0
#             return ts.termination(np.array(self._state, dtype=np.int32), reward)
#         else:
#             return ts.transition(
#                 np.array(self._state, dtype=np.int32), reward=0, discount=0.9)
#
#     def move(self, action):
#         row, col, frow, fcol = self._state[0],self._state[1],self._state[2],self._state[3]
#         if action == 0: #down
#             if row - 1 >= 0:
#                 self._state[0] -= 1
#         if action == 1: #up
#             if row + 1 < 6:
#                 self._state[0] += 1
#         if action == 2: #left
#             if col - 1 >= 0:
#                 self._state[1] -= 1
#         if action == 3: #right
#             if col + 1  < 6:
#                 self._state[1] += 1
#
#     def game_over(self):
#         row, col, frow, fcol = self._state[0],self._state[1],self._state[2],self._state[3]
#         return row==frow and col==fcol
#
# if __name__ == '__main__':
#     env = GridWorldEnv()
#     utils.validate_py_environment(env, episodes=5)
#
#     tl_env = wrappers.TimeLimit(env, duration=50)
#
#     time_step = tl_env.reset()
#     print(time_step)
#     rewards = time_step.reward
#
#     for i in range(100):
#         action = np.random.choice([0,1,2,3])
#         time_step = tl_env.step(action)
#         print(time_step)
#         rewards += time_step.reward
#
#     print(rewards)