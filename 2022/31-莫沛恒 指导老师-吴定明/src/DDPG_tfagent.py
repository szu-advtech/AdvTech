from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt

import tensorflow as tf

from tf_agents.agents.ddpg import ddpg_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment

import configs
from rm_DRL import ClusterEnv
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.td3 import td3_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import py_tf_policy
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


def train_ddpg(
        num_iterations=20000,
        initial_collect_steps=1000,
        collect_steps_per_iteration=1,
        actor_fc_layers=(400, 300),
        critic_obs_fc_layers=(400,),
        critic_action_fc_layers=None,
        critic_joint_fc_layers=(300,),
        replay_buffer_capacity=100000,
        exploration_noise_std=0.1,
        target_update_tau=0.05,
        target_update_period=5,
        actor_update_period=2,
        actor_learning_rate=1e-4,
        critic_learning_rate=1e-3,
        dqda_clipping=None,
        gamma=0.995,
        reward_scale_factor=1.0,
        gradient_clipping=None,
        train_checkpoint_interval=10000,
        policy_checkpoint_interval=5000,
        rb_checkpoint_interval=20000,
        summary_interval=1000,
        debug_summaries=False,
        summarize_grads_and_vars=False,
        eval_metrics_callback=None,
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

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.compat.v1.Variable(0)

    with tf.compat.v2.summary.record_if(
            lambda: tf.math.equal(train_step_counter % summary_interval, 0)):

        actor_net = actor_network.ActorNetwork(
            train_env.time_step_spec().observation,
            train_env.action_spec(),
            fc_layer_params=actor_fc_layers,
        )

        critic_net_input_specs = (train_env.time_step_spec().observation,
                                  train_env.action_spec())

        critic_net = critic_network.CriticNetwork(
            critic_net_input_specs,
            observation_fc_layer_params=critic_obs_fc_layers,
            action_fc_layer_params=critic_action_fc_layers,
            joint_fc_layer_params=critic_joint_fc_layers,
        )

        tf_agent = td3_agent.Td3Agent(
            train_env.time_step_spec(),
            train_env.action_spec(),
            optimizer=optimizer,
            td_errors_loss_fn=common.element_wise_squared_loss,
            actor_network=actor_net,
            critic_network=critic_net,
            actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=actor_learning_rate),
            critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                learning_rate=critic_learning_rate),
            exploration_noise_std=exploration_noise_std,
            target_update_tau=target_update_tau,
            target_update_period=target_update_period,
            actor_update_period=actor_update_period,
            dqda_clipping=dqda_clipping,
            gamma=gamma,
            reward_scale_factor=reward_scale_factor,
            gradient_clipping=gradient_clipping,
            debug_summaries=debug_summaries,
            summarize_grads_and_vars=summarize_grads_and_vars,
            train_step_counter=train_step_counter,
        )

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            tf_agent.collect_data_spec,
            batch_size=train_env.batch_size,
            max_length=replay_buffer_capacity)

        eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)

        train_metrics = [
            tf_metrics.NumberOfEpisodes(),
            tf_metrics.EnvironmentSteps(),
            tf_metrics.AverageReturnMetric(),
            tf_metrics.AverageEpisodeLengthMetric(),
        ]

        collect_policy = tf_agent.collect_policy
        initial_collect_op = dynamic_step_driver.DynamicStepDriver(
            train_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=initial_collect_steps).run()

        collect_op = dynamic_step_driver.DynamicStepDriver(
            train_env,
            collect_policy,
            observers=[replay_buffer.add_batch] + train_metrics,
            num_steps=collect_steps_per_iteration).run()

        collect_data(train_env, tf_agent.collect_policy, replay_buffer, steps=10000)

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=batch_size,
            num_steps=2).prefetch(3)

        iterator = iter(dataset)
        trajectories, unused_info = iterator.get_next()

        train_fn = common.function(tf_agent.train)
        train_op = train_fn(experience=trajectories)

        train_checkpointer = common.Checkpointer(
            tf_agent=tf_agent,
            train_step_counter=train_step_counter,
            metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
        policy_checkpointer = common.Checkpointer(
            policy=tf_agent.policy,
            train_step_counter=train_step_counter)
        rb_checkpointer = common.Checkpointer(
            max_to_keep=1,
            replay_buffer=replay_buffer)

        summary_ops = []
        for train_metric in train_metrics:
            summary_ops.append(train_metric.tf_summaries(
                train_step=train_step_counter, step_metrics=train_metrics[:2]))

        init_agent_op = tf_agent.initialize()

        with tf.compat.v1.Session() as sess:

            train_checkpointer.initialize_or_restore(sess)
            rb_checkpointer.initialize_or_restore(sess)
            sess.run(iterator.initializer)

            common.initialize_uninitialized_variables(sess)

            sess.run(init_agent_op)
            sess.run(initial_collect_op)

            train_step_counter_val = sess.run(train_step_counter)
            metric_utils.compute_summaries(
                eval_env,
                eval_py_policy,
                num_episodes=num_eval_episodes,
                global_step=train_step_counter_val,
                callback=eval_metrics_callback,
                log=True,
            )

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

                if train_step_counter_val % train_checkpoint_interval == 0:
                    train_checkpointer.save(global_step=train_step_counter_val)

                if train_step_counter_val % policy_checkpoint_interval == 0:
                    policy_checkpointer.save(global_step=train_step_counter_val)

                if train_step_counter_val % rb_checkpoint_interval == 0:
                    rb_checkpointer.save(global_step=train_step_counter_val)

                if train_step_counter_val % eval_interval == 0:
                    metric_utils.compute_summaries(
                        eval_py_env,
                        eval_py_policy,
                        num_episodes=num_eval_episodes,
                        global_step=train_step_counter_val,
                        callback=eval_metrics_callback,
                        log=True,
                    )



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
#         num_iterations=2000000,
#         actor_fc_layers=(400, 300),
#         critic_obs_fc_layers=(400,),
#         critic_action_fc_layers=None,
#         critic_joint_fc_layers=(300,),
#         # Params for collect
#         initial_collect_steps=1000,
#         collect_steps_per_iteration=1,
#         replay_buffer_capacity=100000,
#         exploration_noise_std=0.1,
#         # Params for target update
#         target_update_tau=0.05,
#         target_update_period=5,
#         # Params for train
#         train_steps_per_iteration=1,
#         batch_size=64,
#         actor_update_period=2,
#         actor_learning_rate=1e-4,
#         critic_learning_rate=1e-3,
#         dqda_clipping=None,
#         td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
#         gamma=0.995,
#         reward_scale_factor=1.0,
#         gradient_clipping=None,
#         # Params for eval
#         num_eval_episodes=10,
#         eval_interval=10000,
#         # Params for checkpoints, summaries, and logging
#         train_checkpoint_interval=10000,
#         policy_checkpoint_interval=5000,
#         rb_checkpoint_interval=20000,
#         log_interval=1000,
#         summary_interval=1000,
#         summaries_flush_secs=10,
#         debug_summaries=False,
#         summarize_grads_and_vars=False,
#         eval_metrics_callback=None):
#     """A simple train and eval for TD3."""
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
#         py_metrics.AverageReturnMetric(buffer_size=num_eval_episodes),
#         py_metrics.AverageEpisodeLengthMetric(buffer_size=num_eval_episodes),
#     ]
#
#     train_py_env = ClusterEnv()
#     eval_py_env = ClusterEnv()
#     global_step = tf.compat.v1.train.get_or_create_global_step()
#     with tf.compat.v2.summary.record_if(
#             lambda: tf.math.equal(global_step % summary_interval, 0)):
#         tf_env = tf_py_environment.TFPyEnvironment(train_py_env)
#         eval_py_env = tf_py_environment.TFPyEnvironment(eval_py_env)
#
#         actor_net = actor_network.ActorNetwork(
#             tf_env.time_step_spec().observation,
#             tf_env.action_spec(),
#             fc_layer_params=actor_fc_layers,
#         )
#
#         critic_net_input_specs = (tf_env.time_step_spec().observation,
#                                   tf_env.action_spec())
#
#         critic_net = critic_network.CriticNetwork(
#             critic_net_input_specs,
#             observation_fc_layer_params=critic_obs_fc_layers,
#             action_fc_layer_params=critic_action_fc_layers,
#             joint_fc_layer_params=critic_joint_fc_layers,
#         )
#
#         tf_agent = td3_agent.Td3Agent(
#             tf_env.time_step_spec(),
#             tf_env.action_spec(),
#             actor_network=actor_net,
#             critic_network=critic_net,
#             actor_optimizer=tf.compat.v1.train.AdamOptimizer(
#                 learning_rate=actor_learning_rate),
#             critic_optimizer=tf.compat.v1.train.AdamOptimizer(
#                 learning_rate=critic_learning_rate),
#             exploration_noise_std=exploration_noise_std,
#             target_update_tau=target_update_tau,
#             target_update_period=target_update_period,
#             actor_update_period=actor_update_period,
#             dqda_clipping=dqda_clipping,
#             td_errors_loss_fn=td_errors_loss_fn,
#             gamma=gamma,
#             reward_scale_factor=reward_scale_factor,
#             gradient_clipping=gradient_clipping,
#             debug_summaries=debug_summaries,
#             summarize_grads_and_vars=summarize_grads_and_vars,
#             train_step_counter=global_step,
#         )
#
#         replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
#             tf_agent.collect_data_spec,
#             batch_size=tf_env.batch_size,
#             max_length=replay_buffer_capacity)
#
#         eval_py_policy = py_tf_policy.PyTFPolicy(tf_agent.policy)
#
#         train_metrics = [
#             tf_metrics.NumberOfEpisodes(),
#             tf_metrics.EnvironmentSteps(),
#             tf_metrics.AverageReturnMetric(),
#             tf_metrics.AverageEpisodeLengthMetric(),
#         ]
#
#         collect_policy = tf_agent.collect_policy
#         initial_collect_op = dynamic_step_driver.DynamicStepDriver(
#             tf_env,
#             collect_policy,
#             observers=[replay_buffer.add_batch] + train_metrics,
#             num_steps=initial_collect_steps).run()
#
#         collect_op = dynamic_step_driver.DynamicStepDriver(
#             tf_env,
#             collect_policy,
#             observers=[replay_buffer.add_batch] + train_metrics,
#             num_steps=collect_steps_per_iteration).run()
#
#         dataset = replay_buffer.as_dataset(
#             num_parallel_calls=3,
#             sample_batch_size=batch_size,
#             num_steps=2).prefetch(3)
#         iterator = tf.compat.v1.data.make_initializable_iterator(dataset)
#         trajectories, unused_info = iterator.get_next()
#
#         train_fn = common.function(tf_agent.train)
#         train_op = train_fn(experience=trajectories)
#
#         train_checkpointer = common.Checkpointer(
#             ckpt_dir=train_dir,
#             agent=tf_agent,
#             global_step=global_step,
#             metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
#         policy_checkpointer = common.Checkpointer(
#             ckpt_dir=os.path.join(train_dir, 'policy'),
#             policy=tf_agent.policy,
#             global_step=global_step)
#         rb_checkpointer = common.Checkpointer(
#             ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
#             max_to_keep=1,
#             replay_buffer=replay_buffer)
#
#         summary_ops = []
#         for train_metric in train_metrics:
#             summary_ops.append(train_metric.tf_summaries(
#                 train_step=global_step, step_metrics=train_metrics[:2]))
#
#         with eval_summary_writer.as_default(), \
#              tf.compat.v2.summary.record_if(True):
#             for eval_metric in eval_metrics:
#                 eval_metric.tf_summaries(train_step=global_step)
#
#         init_agent_op = tf_agent.initialize()
#
#         with tf.compat.v1.Session() as sess:
#             # Initialize the graph.
#             train_checkpointer.initialize_or_restore(sess)
#             rb_checkpointer.initialize_or_restore(sess)
#             sess.run(iterator.initializer)

#             common.initialize_uninitialized_variables(sess)
#
#             sess.run(init_agent_op)
#             sess.run(train_summary_writer.init())
#             sess.run(eval_summary_writer.init())
#             sess.run(initial_collect_op)
#
#             global_step_val = sess.run(global_step)
#             metric_utils.compute_summaries(
#                 eval_metrics,
#                 eval_py_env,
#                 eval_py_policy,
#                 num_episodes=num_eval_episodes,
#                 global_step=global_step_val,
#                 callback=eval_metrics_callback,
#                 log=True,
#             )
#
#             collect_call = sess.make_callable(collect_op)
#             train_step_call = sess.make_callable([train_op, summary_ops, global_step])
#
#             timed_at_step = sess.run(global_step)
#             time_acc = 0
#             steps_per_second_ph = tf.compat.v1.placeholder(
#                 tf.float32, shape=(), name='steps_per_sec_ph')
#             steps_per_second_summary = tf.compat.v2.summary.scalar(
#                 name='global_steps_per_sec', data=steps_per_second_ph,
#                 step=global_step)
#
#             for _ in range(num_iterations):
#                 start_time = time.time()
#                 collect_call()
#                 for _ in range(train_steps_per_iteration):
#                     loss_info_value, _, global_step_val = train_step_call()
#                 time_acc += time.time() - start_time
#
#                 if global_step_val % log_interval == 0:
#                     logging.info('step = %d, loss = %f', global_step_val,
#                                  loss_info_value.loss)
#                     steps_per_sec = (global_step_val - timed_at_step) / time_acc
#                     logging.info('%.3f steps/sec', steps_per_sec)
#                     sess.run(
#                         steps_per_second_summary,
#                         feed_dict={steps_per_second_ph: steps_per_sec})
#                     timed_at_step = global_step_val
#                     time_acc = 0
#
#                 if global_step_val % train_checkpoint_interval == 0:
#                     train_checkpointer.save(global_step=global_step_val)
#
#                 if global_step_val % policy_checkpoint_interval == 0:
#                     policy_checkpointer.save(global_step=global_step_val)
#
#                 if global_step_val % rb_checkpoint_interval == 0:
#                     rb_checkpointer.save(global_step=global_step_val)
#
#                 if global_step_val % eval_interval == 0:
#                     metric_utils.compute_summaries(
#                         eval_metrics,
#                         eval_py_env,
#                         eval_py_policy,
#                         num_episodes=num_eval_episodes,
#                         global_step=global_step_val,
#                         callback=eval_metrics_callback,
#                         log=True,
#                     )
#
#
# def main(_):
#     logging.set_verbosity(logging.INFO)
#     tf.compat.v1.enable_resource_variables()
#     train_eval('/home/tawfiq/Desktop/RL', num_iterations=10000)
#
#
# if __name__ == '__main__':
#     #flags.mark_flag_as_required('root_dir')
#     app.run(main)
