from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.ppo import ppo_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
import time
import configs
from rm_DRL import ClusterEnv
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.agents.ppo import ppo_clip_agent
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network
from tf_agents.policies import policy_saver
from tf_agents.replay_buffers import tf_uniform_replay_buffer
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


def train_ppo(

        num_iterations=20000,
        initial_collect_steps=1000,
        collect_steps_per_iteration=1,
        replay_buffer_max_length=100000,
        fc_layer_params=(200,),
        batch_size=64,
        learning_rate=1e-3,
        log_interval=200,
        num_eval_episodes=10,
        eval_interval=1000,
        actor_fc_layers=(200, 100),
        value_fc_layers=(200, 100),
        use_rnns=False,
        num_environment_steps=25000000,
        collect_episodes_per_iteration=30,
        num_parallel_environments=30,
        replay_buffer_capacity=1001, 
        num_epochs=25,
        use_tf_functions=True,
        debug_summaries=False,
        summarize_grads_and_vars=False
):
    file = open(configs.root + '/output/avg_returns_' + configs.algo + '_beta_' + str(configs.beta) + '.csv', 'w',
                newline='')
    avg_return_writer = csv.writer(file, delimiter=',')
    avg_return_writer.writerow(["Iteration", "AVG_Return"])

    train_py_env = ClusterEnv()
    eval_py_env = train_py_env

    train_env = tf_py_environment.TFPyEnvironment(train_py_env)
    eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)

    if use_rnns:
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            input_fc_layer_params=actor_fc_layers,
            output_fc_layer_params=None)
        value_net = value_rnn_network.ValueRnnNetwork(
            train_env.observation_spec(),
            input_fc_layer_params=value_fc_layers,
            output_fc_layer_params=None)
    else:
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            train_env.observation_spec(),
            train_env.action_spec(),
            fc_layer_params=actor_fc_layers,
            activation_fn=tf.keras.activations.tanh)
        value_net = value_network.ValueNetwork(
            train_env.observation_spec(),
            fc_layer_params=value_fc_layers,
            activation_fn=tf.keras.activations.tanh)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v1.Variable(0)

    tf_agent = ppo_clip_agent.PPOClipAgent(
        train_env.time_step_spec(),
        train_env.action_spec(),
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        actor_net=actor_net,
        value_net=value_net,
        entropy_regularization=0.0,
        importance_ratio_clipping=0.2,
        normalize_observations=False,
        normalize_rewards=False,
        use_gae=True,
        num_epochs=num_epochs,
        debug_summaries=debug_summaries,
        summarize_grads_and_vars=summarize_grads_and_vars,
        train_step_counter=train_step_counter)
    tf_agent.initialize()

    environment_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        environment_steps_metric,
    ]

    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(
            batch_size=num_parallel_environments),
        tf_metrics.AverageEpisodeLengthMetric(
            batch_size=num_parallel_environments),
    ]

    eval_policy = tf_agent.policy
    collect_policy = tf_agent.collect_policy

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        tf_agent.collect_data_spec,
        batch_size=num_parallel_environments,
        max_length=replay_buffer_capacity)

    collect_data(train_env, tf_agent.collect_policy, replay_buffer, steps=10000)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=batch_size,
        num_steps=2).prefetch(3)

    iterator = iter(dataset)

    train_checkpointer = common.Checkpointer(
        agent=tf_agent,
        train_step_counter=train_step_counter,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    policy_checkpointer = common.Checkpointer(
        policy=eval_policy,
        train_step_counter=train_step_counter)
    saved_model = policy_saver.PolicySaver(
        eval_policy, train_step=train_step_counter)

    train_checkpointer.initialize_or_restore()

    collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
        train_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_episodes=collect_episodes_per_iteration)

    def train_step():
        trajectories = replay_buffer.gather_all()
        return tf_agent.train(experience=trajectories)

    if use_tf_functions:
        collect_driver.run = common.function(collect_driver.run, autograph=False)
        tf_agent.train = common.function(tf_agent.train, autograph=False)
        train_step = common.function(train_step)

    collect_time = 0
    train_time = 0
    timed_at_step = train_step_counter.numpy()

    while environment_steps_metric.result() < num_environment_steps:
        train_step_counter_val = train_step_counter.numpy()
        if train_step_counter_val % eval_interval == 0:
            metric_utils.eager_compute(
                eval_env,
                eval_policy,
                num_episodes=num_eval_episodes,
                train_step=train_step_counter,
                summary_prefix='Metrics',
            )

    start_time = time.time()
    collect_driver.run()
    collect_time += time.time() - start_time

    start_time = time.time()
    total_loss, _ = train_step()
    replay_buffer.clear()
    train_time += time.time() - start_time

    for train_metric in train_metrics:
        train_metric.tf_summaries(
            train_step=train_step_counter, step_metrics=step_metrics)

    tf_agent.train = common.function(tf_agent.train)

    tf_agent.train_step_counter.assign(0)

    avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, tf_agent.collect_policy, replay_buffer)

        experience, unused_info = next(iterator)
        train_loss = tf_agent.train(experience).loss

        step = tf_agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, tf_agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            avg_return_writer.writerow([step, avg_return])
            returns.append(avg_return)


    # *** Visualizations ***
    # iterations = range(0, num_iterations + 1, eval_interval)
    # plt.plot(iterations, returns)
    # plt.ylabel('Average Return')
    # plt.xlabel('Iterations')
    # # plt.ylim(top=250)
    # plt.show()



    # def train_eval(
    #         root_dir,
    #         env_name='HalfCheetah-v2',
    #         env_load_fn=suite_mujoco.load,
    #         random_seed=None,

    #         actor_fc_layers=(200, 100),
    #         value_fc_layers=(200, 100),
    #         use_rnns=False,
    #         # Params for collect
    #         num_environment_steps=25000000,
    #         collect_episodes_per_iteration=30,
    #         num_parallel_environments=30,
    #         replay_buffer_capacity=1001,  # Per-environment
    #         # Params for train
    #         num_epochs=25,
    #         learning_rate=1e-3,
    #         # Params for eval
    #         num_eval_episodes=30,
    #         eval_interval=500,
    #         # Params for summaries and logging
    #         train_checkpoint_interval=500,
    #         policy_checkpoint_interval=500,
    #         log_interval=50,
    #         summary_interval=50,
    #         summaries_flush_secs=1,
    #         use_tf_functions=True,
    #         debug_summaries=False,
    #         summarize_grads_and_vars=False):
    #     """A simple train and eval for PPO."""
    #     if root_dir is None:
    #         raise AttributeError('train_eval requires a root_dir.')
    #
    #     root_dir = os.path.expanduser(root_dir)
    #     train_dir = os.path.join(root_dir, 'train')
    #     eval_dir = os.path.join(root_dir, 'eval')
    #     saved_model_dir = os.path.join(root_dir, 'policy_saved_model')
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
    #         if random_seed is not None:
    #             tf.compat.v1.set_random_seed(random_seed)
    #         eval_tf_env = tf_py_environment.TFPyEnvironment(env_load_fn(env_name))
    #         tf_env = tf_py_environment.TFPyEnvironment(
    #             parallel_py_environment.ParallelPyEnvironment(
    #                 [lambda: env_load_fn(env_name)] * num_parallel_environments))
    #         optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    #
    #         if use_rnns:
    #             actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
    #                 tf_env.observation_spec(),
    #                 tf_env.action_spec(),
    #                 input_fc_layer_params=actor_fc_layers,
    #                 output_fc_layer_params=None)
    #             value_net = value_rnn_network.ValueRnnNetwork(
    #                 tf_env.observation_spec(),
    #                 input_fc_layer_params=value_fc_layers,
    #                 output_fc_layer_params=None)
    #         else:
    #             actor_net = actor_distribution_network.ActorDistributionNetwork(
    #                 tf_env.observation_spec(),
    #                 tf_env.action_spec(),
    #                 fc_layer_params=actor_fc_layers,
    #                 activation_fn=tf.keras.activations.tanh)
    #             value_net = value_network.ValueNetwork(
    #                 tf_env.observation_spec(),
    #                 fc_layer_params=value_fc_layers,
    #                 activation_fn=tf.keras.activations.tanh)
    #
    #         tf_agent = ppo_clip_agent.PPOClipAgent(
    #             tf_env.time_step_spec(),
    #             tf_env.action_spec(),
    #             optimizer,
    #             actor_net=actor_net,
    #             value_net=value_net,
    #             entropy_regularization=0.0,
    #             importance_ratio_clipping=0.2,
    #             normalize_observations=False,
    #             normalize_rewards=False,
    #             use_gae=True,
    #             num_epochs=num_epochs,
    #             debug_summaries=debug_summaries,
    #             summarize_grads_and_vars=summarize_grads_and_vars,
    #             train_step_counter=global_step)
    #         tf_agent.initialize()
    #
    #         environment_steps_metric = tf_metrics.EnvironmentSteps()
    #         step_metrics = [
    #             tf_metrics.NumberOfEpisodes(),
    #             environment_steps_metric,
    #         ]
    #
    #         train_metrics = step_metrics + [
    #             tf_metrics.AverageReturnMetric(
    #                 batch_size=num_parallel_environments),
    #             tf_metrics.AverageEpisodeLengthMetric(
    #                 batch_size=num_parallel_environments),
    #         ]
    #
    #         eval_policy = tf_agent.policy
    #         collect_policy = tf_agent.collect_policy
    #
    #         replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    #             tf_agent.collect_data_spec,
    #             batch_size=num_parallel_environments,
    #             max_length=replay_buffer_capacity)
    #
    #         train_checkpointer = common.Checkpointer(
    #             ckpt_dir=train_dir,
    #             agent=tf_agent,
    #             global_step=global_step,
    #             metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
    #         policy_checkpointer = common.Checkpointer(
    #             ckpt_dir=os.path.join(train_dir, 'policy'),
    #             policy=eval_policy,
    #             global_step=global_step)
    #         saved_model = policy_saver.PolicySaver(
    #             eval_policy, train_step=global_step)
    #
    #         train_checkpointer.initialize_or_restore()
    #
    #         collect_driver = dynamic_episode_driver.DynamicEpisodeDriver(
    #             tf_env,
    #             collect_policy,
    #             observers=[replay_buffer.add_batch] + train_metrics,
    #             num_episodes=collect_episodes_per_iteration)
    #
    #         def train_step():
    #             trajectories = replay_buffer.gather_all()
    #             return tf_agent.train(experience=trajectories)
    #
    #         if use_tf_functions:

    #             collect_driver.run = common.function(collect_driver.run, autograph=False)
    #             tf_agent.train = common.function(tf_agent.train, autograph=False)
    #             train_step = common.function(train_step)
    #
    #         collect_time = 0
    #         train_time = 0
    #         timed_at_step = global_step.numpy()
    #
    #         while environment_steps_metric.result() < num_environment_steps:
    #             global_step_val = global_step.numpy()
    #             if global_step_val % eval_interval == 0:
    #                 metric_utils.eager_compute(
    #                     eval_metrics,
    #                     eval_tf_env,
    #                     eval_policy,
    #                     num_episodes=num_eval_episodes,
    #                     train_step=global_step,
    #                     summary_writer=eval_summary_writer,
    #                     summary_prefix='Metrics',
    #                 )
    #
    #             start_time = time.time()
    #             collect_driver.run()
    #             collect_time += time.time() - start_time
    #
    #             start_time = time.time()
    #             total_loss, _ = train_step()
    #             replay_buffer.clear()
    #             train_time += time.time() - start_time
    #
    #             for train_metric in train_metrics:
    #                 train_metric.tf_summaries(
    #                     train_step=global_step, step_metrics=step_metrics)
    #
    #             if global_step_val % log_interval == 0:
    #                 logging.info('step = %d, loss = %f', global_step_val, total_loss)
    #                 steps_per_sec = (
    #                         (global_step_val - timed_at_step) / (collect_time + train_time))
    #                 logging.info('%.3f steps/sec', steps_per_sec)
    #                 logging.info('collect_time = %.3f, train_time = %.3f', collect_time,
    #                              train_time)
    #                 with tf.compat.v2.summary.record_if(True):
    #                     tf.compat.v2.summary.scalar(
    #                         name='global_steps_per_sec', data=steps_per_sec, step=global_step)
    #
    #                 if global_step_val % train_checkpoint_interval == 0:
    #                     train_checkpointer.save(global_step=global_step_val)
    #
    #                 if global_step_val % policy_checkpoint_interval == 0:
    #                     policy_checkpointer.save(global_step=global_step_val)
    #                     saved_model_path = os.path.join(
    #                         saved_model_dir, 'policy_' + ('%d' % global_step_val).zfill(9))
    #                     saved_model.save(saved_model_path)
    #
    #                 timed_at_step = global_step_val
    #                 collect_time = 0
    #                 train_time = 0
    #
    #         # One final eval before exiting.
    #         metric_utils.eager_compute(
    #             eval_metrics,
    #             eval_tf_env,
    #             eval_policy,
    #             num_episodes=num_eval_episodes,
    #             train_step=global_step,
    #             summary_writer=eval_summary_writer,
    #             summary_prefix='Metrics',
    #         )
    #
    # def main(_):
    #     logging.set_verbosity(logging.INFO)
    #     tf.compat.v1.enable_v2_behavior()
    #     train_eval(
    #         FLAGS.root_dir,
    #         env_name=FLAGS.env_name,
    #         use_rnns=FLAGS.use_rnns,
    #         num_environment_steps=FLAGS.num_environment_steps,
    #         collect_episodes_per_iteration=FLAGS.collect_episodes_per_iteration,
    #         num_parallel_environments=FLAGS.num_parallel_environments,
    #         replay_buffer_capacity=FLAGS.replay_buffer_capacity,
    #         num_epochs=FLAGS.num_epochs,
    #         num_eval_episodes=FLAGS.num_eval_episodes)
    #
    # if __name__ == '__main__':
    #     flags.mark_flag_as_required('root_dir')
    #     multiprocessing.handle_main(lambda _: app.run(main))