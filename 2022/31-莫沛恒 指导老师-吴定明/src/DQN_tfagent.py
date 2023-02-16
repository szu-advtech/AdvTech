from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment

import configs
from rm_DRL import ClusterEnv
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import csv
tf.compat.v1.enable_v2_behavior()


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


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    for _ in range(steps):
        collect_step(env, policy, buffer)


def train_dqn(
        num_iterations=20000,
        initial_collect_steps=1000,
        collect_steps_per_iteration=1,
        replay_buffer_max_length=100000,
        fc_layer_params=(200,),
        batch_size=64,
        learning_rate=1e-3,
        log_interval=200,
        num_eval_episodes=10,
        eval_interval=1000
):
    file = open(configs.root + '/output/avg_returns_' + configs.algo + '_beta_' + str(configs.beta) + '.csv', 'w',
                newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return"])

    train_py_env = ClusterEnv()
    eval_py_env = train_py_env

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    q_net = q_network.QNetwork(
        train_env.observation_spec(),
        train_env.action_spec(),
        fc_layer_params=fc_layer_params)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v1.Variable(0)

    agent = dqn_agent.DqnAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    collect_data(train_env, agent.collect_policy, replay_buffer, steps=10000)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    agent.train = common.function(agent.train)

    agent.train_step_counter.assign(0)

    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        experience, unused_info = next(iterator)
        train_loss = agent.train(experience).loss

        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            avg_return_writer.writerow([step, avg_return])
            returns.append(avg_return)



# def train_eval(
#         root_dir,
#         env_name='CartPole-v0',
#         num_iterations=100000,
#         train_sequence_length=1,
#         # Params for QNetwork
#         fc_layer_params=(100,),
#         # Params for QRnnNetwork
#         input_fc_layer_params=(50,),
#         lstm_size=(20,),
#         output_fc_layer_params=(20,),
#
#         # Params for collect
#         initial_collect_steps=1000,
#         collect_steps_per_iteration=1,
#         epsilon_greedy=0.1,
#         replay_buffer_capacity=100000,
#         # Params for target update
#         target_update_tau=0.05,
#         target_update_period=5,
#         # Params for train
#         train_steps_per_iteration=1,
#         batch_size=64,
#         learning_rate=1e-3,
#         n_step_update=1,
#         gamma=0.99,
#         reward_scale_factor=1.0,
#         gradient_clipping=None,
#         use_tf_functions=True,
#         # Params for eval
#         num_eval_episodes=10,
#         eval_interval=1000,
#         # Params for checkpoints
#         train_checkpoint_interval=10000,
#         policy_checkpoint_interval=5000,
#         rb_checkpoint_interval=20000,
#         # Params for summaries and logging
#         log_interval=1000,
#         summary_interval=1000,
#         summaries_flush_secs=10,
#         debug_summaries=False,
#         summarize_grads_and_vars=False,
#         eval_metrics_callback=None):
#     """A simple train and eval for DQN."""
#     root_dir = os.path.expanduser(root_dir)
#     train_dir = os.path.join(root_dir, 'train')
#     eval_dir = os.path.join(root_dir, 'eval')
#
#     train_summary_writer = tf.compat.v2.summary.create_file_writer(
#         train_dir, flush_millis=summaries_flush_secs * 1000)
#     train_summary_writer.set_as_default()
#
#     eval_summary_writer = tf.compat.v2.summary.create_file_writer(
#         eval_dir, flush_millis=summaries_flush_secs * 1000)
#     eval_metrics = [
#         tf_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
#         tf_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes)
#     ]
#
#     global_step = tf.compat.v1.train.get_or_create_global_step()
#     with tf.compat.v2.summary.record_if(
#             lambda: tf.math.equal(global_step % summary_interval, 0)):
#         tf_env = tf_py_environment.TFPyEnvironment(ClusterEnv())
#         eval_tf_env = tf_py_environment.TFPyEnvironment(ClusterEnv())
#
#         if train_sequence_length != 1 and n_step_update != 1:
#             raise NotImplementedError(
#                 'train_eval does not currently support n-step updates with stateful '
#                 'networks (i.e., RNNs)')
#
#         if train_sequence_length > 1:
#             q_net = q_rnn_network.QRnnNetwork(
#                 tf_env.observation_spec(),
#                 tf_env.action_spec(),
#                 input_fc_layer_params=input_fc_layer_params,
#                 lstm_size=lstm_size,
#                 output_fc_layer_params=output_fc_layer_params)
#         else:
#             q_net = q_network.QNetwork(
#                 tf_env.observation_spec(),
#                 tf_env.action_spec(),
#                 fc_layer_params=fc_layer_params)
#             train_sequence_length = n_step_update
#
#         tf_agent = dqn_agent.DqnAgent(
#             tf_env.time_step_spec(),
#             tf_env.action_spec(),
#             q_network=q_net,
#             epsilon_greedy=epsilon_greedy,
#             n_step_update=n_step_update,
#             target_update_tau=target_update_tau,
#             target_update_period=target_update_period,
#             optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate),
#             td_errors_loss_fn=common.element_wise_squared_loss,
#             gamma=gamma,
#             reward_scale_factor=reward_scale_factor,
#             gradient_clipping=gradient_clipping,
#             debug_summaries=debug_summaries,
#             summarize_grads_and_vars=summarize_grads_and_vars,
#             train_step_counter=global_step)
#         tf_agent.initialize()
#
#         train_metrics = [
#             tf_metrics.NumberOfEpisodes(),
#             tf_metrics.EnvironmentSteps(),
#             tf_metrics.AverageReturnMetric(),
#             tf_metrics.AverageEpisodeLengthMetric(),
#         ]
#
#         eval_policy = tf_agent.policy
#         collect_policy = tf_agent.collect_policy
#
#         replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#             data_spec=tf_agent.collect_data_spec,
#             batch_size=tf_env.batch_size,
#             max_length=replay_buffer_capacity)
#
#         collect_driver = dynamic_step_driver.DynamicStepDriver(
#             tf_env,
#             collect_policy,
#             observers=[replay_buffer.add_batch] + train_metrics,
#             num_steps=collect_steps_per_iteration)
#         print('i was here \n')
#         train_checkpointer = common.Checkpointer(
#             ckpt_dir=train_dir,
#             agent=tf_agent,
#             global_step=global_step,
#             metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
#         print('i was here too!\n')
#         policy_checkpointer = common.Checkpointer(
#             ckpt_dir=os.path.join(train_dir, 'policy'),
#             policy=eval_policy,
#             global_step=global_step)
#         rb_checkpointer = common.Checkpointer(
#             ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
#             max_to_keep=1,
#             replay_buffer=replay_buffer)
#
#         train_checkpointer.initialize_or_restore()
#         rb_checkpointer.initialize_or_restore()
#
#         if use_tf_functions:
#             # To speed up collect use common.function.
#             collect_driver.run = common.function(collect_driver.run)
#             tf_agent.train = common.function(tf_agent.train)
#
#         initial_collect_policy = random_tf_policy.RandomTFPolicy(
#             tf_env.time_step_spec(), tf_env.action_spec())
#
#         # Collect initial replay data.
#         logging.info(
#             'Initializing replay buffer by collecting experience for %d steps with '
#             'a random policy.', initial_collect_steps)
#         dynamic_step_driver.DynamicStepDriver(
#             tf_env,
#             initial_collect_policy,
#             observers=[replay_buffer.add_batch] + train_metrics,
#             num_steps=initial_collect_steps).run()
#
#         results = metric_utils.eager_compute(
#             eval_metrics,
#             eval_tf_env,
#             eval_policy,
#             num_episodes=num_eval_episodes,
#             train_step=global_step,
#             summary_writer=eval_summary_writer,
#             summary_prefix='Metrics',
#         )
#         if eval_metrics_callback is not None:
#             eval_metrics_callback(results, global_step.numpy())
#         metric_utils.log_metrics(eval_metrics)
#
#         time_step = None
#         policy_state = collect_policy.get_initial_state(tf_env.batch_size)
#
#         timed_at_step = global_step.numpy()
#         time_acc = 0
#
#         # Dataset generates trajectories with shape [Bx2x...]
#         dataset = replay_buffer.as_dataset(
#             num_parallel_calls=3,
#             sample_batch_size=batch_size,
#             num_steps=train_sequence_length + 1).prefetch(3)
#         iterator = iter(dataset)
#
#         def train_step():
#             experience, _ = next(iterator)
#             return tf_agent.train(experience)
#
#         if use_tf_functions:
#             train_step = common.function(train_step)
#
#         for _ in range(num_iterations):
#             start_time = time.time()
#             time_step, policy_state = collect_driver.run(
#                 time_step=time_step,
#                 policy_state=policy_state,
#             )
#             for _ in range(train_steps_per_iteration):
#                 train_loss = train_step()
#             time_acc += time.time() - start_time
#
#             if global_step.numpy() % log_interval == 0:
#                 logging.info('step = %d, loss = %f', global_step.numpy(),
#                              train_loss.loss)
#                 steps_per_sec = (global_step.numpy() - timed_at_step) / time_acc
#                 logging.info('%.3f steps/sec', steps_per_sec)
#                 tf.compat.v2.summary.scalar(
#                     name='global_steps_per_sec', data=steps_per_sec, step=global_step)
#                 timed_at_step = global_step.numpy()
#                 time_acc = 0
#
#             for train_metric in train_metrics:
#                 train_metric.tf_summaries(
#                     train_step=global_step, step_metrics=train_metrics[:2])
#
#             if global_step.numpy() % train_checkpoint_interval == 0:
#                 train_checkpointer.save(global_step=global_step.numpy())
#
#             if global_step.numpy() % policy_checkpoint_interval == 0:
#                 policy_checkpointer.save(global_step=global_step.numpy())
#
#             if global_step.numpy() % rb_checkpoint_interval == 0:
#                 rb_checkpointer.save(global_step=global_step.numpy())
#
#             if global_step.numpy() % eval_interval == 0:
#                 results = metric_utils.eager_compute(
#                     eval_metrics,
#                     eval_tf_env,
#                     eval_policy,
#                     num_episodes=num_eval_episodes,
#                     train_step=global_step,
#                     summary_writer=eval_summary_writer,
#                     summary_prefix='Metrics',
#                 )
#                 if eval_metrics_callback is not None:
#                     eval_metrics_callback(results, global_step.numpy())
#                 metric_utils.log_metrics(eval_metrics)
#         return train_loss
#
#
# def main(_):
#     logging.set_verbosity(logging.INFO)
#     tf.compat.v1.enable_v2_behavior()
#     # gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
#     train_eval("/home/tawfiq/testRL", num_iterations=FLAGS.num_iterations)
#
#
# if __name__ == '__main__':
#     # flags.mark_flag_as_required('root_dir')
#     app.run(main)
