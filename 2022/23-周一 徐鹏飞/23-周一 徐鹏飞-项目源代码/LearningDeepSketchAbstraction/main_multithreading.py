import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import logging
import datetime  # logging training time

import abstraction_process
import dataset
import abs_utils
import classifier_config
from agent import Agent
from agent import discnt_rwd
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import platform
import tensorflow as tf
# if platform.system() == "Windows":  # for my machine
#     pass
# else:  # for the server
#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# note: keep or remove 0padded ss
from environment import *

__epoch_shuffle_seeds__ = abs_utils.epoch_shuffle_seed(rl_config.shuffle_seed(), classifier_config.epoch())
__epoch_shuffle_seeds_idx__: int = 0


__EVALUATION_MODE__: bool = False
__OUTPUT_LOGS__: bool = False

executor = ThreadPoolExecutor()
selected_classifier_model_path: str = "./training/epoch_4_eacc89" # classifier_model_path() + "/on_auth_preprocessed/epoch_11_8987"
shared_classifier: models.Model = None
# print("classifier model: ")
# print(shared_classifier.summary())

writer = tf.summary.create_file_writer(logdir="./tblog")

action_probs_record_filepath: str = './action_prob_record'

step_global: int = 0
s_cnt: int = 0
loss_cnt: int = 0


def train_multithreading(agent: Agent, save_every_s_episode: int = 1000):
    global __epoch_shuffle_seeds__, __epoch_shuffle_seeds_idx__
    __epoch_shuffle_seeds__ = abs_utils.epoch_shuffle_seed(rl_config.shuffle_seed(), rl_config.epoch())
    __epoch_shuffle_seeds_idx__ = 0
    batch_size: int = rl_config.batch_size()
    sketch_per_thread = rl_config.sketch_per_thread()
    assert batch_size >= sketch_per_thread
    thread_num: int = batch_size // sketch_per_thread

    task_env = []
    for _ in range(thread_num):
        task_env.append(SAEnv(shared_classifier=shared_classifier))

    # logger config
    logging.basicConfig(filename="rl_train.log",
                        level=logging.INFO,
                        format='%(message)s')

    def play(sketches_np, labels_gt, env: SAEnv):
        S_thread = []
        A_thread = []
        R_thread = []
        total_rewards_thread = []
        removed_strokes_idx_thread = []

        for sketch_np, label_gt in zip(sketches_np, labels_gt):
            env.reset()

            # Un-preprocessed copy of sketch_np data,
            # used to index action_process_thread
            sketch_np_ori = copy.deepcopy(sketch_np)

            sketch_np = env.set_cur_sketch(sketch_data=sketch_np, label_gt=label_gt)

            S, A, R = [], [], []
            rm_ss_idx = []

            total_reward: float = 0.
            done = False
            ss_idx: int = 0
            while not done:
                action = agent.select_action(sketch_np, ss_idx)

                sketch_next_np, reward, done, info = env.step(action)
                total_reward += reward

                S.append(sketch_np)
                A.append(action)
                R.append(reward)

                # tensorboard
                global step_global, s_cnt
                with writer.as_default():
                    tf.summary.scalar(name="action", data=action, step=step_global)
                    tf.summary.scalar(name="reward", data=reward, step=step_global)
                    if info is not None:
                        tf.summary.scalar(name="b_r", data=info["b_r"], step=step_global)
                        tf.summary.scalar(name="r_r", data=info["r_r"], step=step_global)
                    if done:
                        tf.summary.scalar(name="last_step_reward", data=reward, step=s_cnt)
                        tf.summary.scalar(name="return", data=discnt_rwd(R)[0][0], step=s_cnt)
                step_global += 1
                s_cnt += 1

                if action == 0:
                    rm_ss_idx.append(ss_idx)
                ss_idx += 1

            S_thread.append(np.array(S))
            A_thread.append(np.array(A))
            R_thread.append(np.array(R))
            total_rewards_thread.append(total_reward)
            removed_strokes_idx_thread.append(((sketch_np_ori, label_gt), rm_ss_idx))

        return S_thread, A_thread, R_thread, total_rewards_thread, removed_strokes_idx_thread

    for e in range(rl_config.epoch()):
        logging.info("{} epoch_{}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), e))

        dataset.shuffle_train_data(__epoch_shuffle_seeds__[__epoch_shuffle_seeds_idx__])
        __epoch_shuffle_seeds_idx__ += 1

        num_episodes: int = \
            len(dataset.categories) * rl_config.num_training_per_category() // rl_config.batch_size()

        abs_prcs_eps = {}

        a_p_batch = []

        for episode in range(num_episodes):
            start: int = batch_size * episode
            end: int = batch_size * (episode + 1)

            S_batch = []
            A_batch = []
            R_batch = []
            total_rewards_batch = []
            tasks_batch = []

            i: int = start
            j: int = 0

            while i < end:
                task = executor.submit(play,
                                       sketches_np=dataset.__train_data_list__[i: min(end, i + sketch_per_thread)],
                                       labels_gt=dataset.__train_label_list__[i: min(end, i + sketch_per_thread)],
                                       env=task_env[j])
                tasks_batch.append(task)
                i += sketch_per_thread
                j += 1

            for task in as_completed(tasks_batch):
                S_task, A_task, R_task, total_rewards_task, removed_sidx_task = task.result()
                S_batch.extend(S_task)
                A_batch.extend(A_task)
                R_batch.extend(R_task)
                total_rewards_batch.extend(total_rewards_task)

                # store episodic abstraction process
                if episode not in abs_prcs_eps:
                    abs_prcs_eps[episode] = []

                abs_prcs_eps[episode].extend(removed_sidx_task)

            # Fit abstraction agent with `POLICY_GRADIENT`
            # loss, action_probs = agent.fit(S_batch, A_batch, R_batch)
            # action_probs = agent.fit(S_batch, A_batch, R_batch)
            act_probs, losses = agent.fit(S_batch, A_batch, R_batch)
            # compute keep remove ratio
            global loss_cnt
            with writer.as_default():
                assert len(act_probs) == len(losses)
                for i in range(len(losses)):
                    tf.summary.scalar(name="loss", data=losses[i], step=loss_cnt)
                    tf.summary.scalar(name="r_p", data=act_probs[i][0], step=loss_cnt)
                    tf.summary.scalar(name="k_p", data=act_probs[i][1], step=loss_cnt)
                    loss_cnt += 1


            # note: put in tensorboard inst
            # msg: str = "{}/{}: {} r_avg: {}".format(
            #     episode + 1, num_episodes,
            #     total_rewards_batch, sum(total_rewards_batch) / batch_size,
            #     # loss
            # )
            msg: str = "{}/{}".format(episode + 1, num_episodes)
            print(msg)
            # logging.info(msg)

            # a_p_batch.extend(action_probs)
            # with writer.as_default():
            #     tf.summary.scalar('p_r', )
            # print(a_p_batch)
            if __OUTPUT_LOGS__ and episode % save_every_s_episode == 0 and episode != 0:
                save_agent_path: str = rl_config.agent_model_path() + "/epoch_{}_ep{}".format(e, episode)
                if rl_config.__REDUCE_LOADED_DATA__ is True:
                    save_agent_path = save_agent_path + "_reduced"
                agent.model.save(filepath=save_agent_path, overwrite=True, include_optimizer=True)
                print("Episodic save: Agent model saved to", save_agent_path)

                # to prevent abstraction process copy, do not use multithreading / multiprocessing
                # It should be acceptable to pause the training progress every 1000 episodes, to save abstraction process
                abstraction_process.write_abs_prcs(abs_prcs_eps)
                print("Saved episodic abstraction process.")
                abs_prcs_eps.clear()

                with open(action_probs_record_filepath, 'a') as f:
                    for a_p in a_p_batch:
                        f.write("{} {}\n".format(a_p[0][0], a_p[0][1]))

                a_p_batch.clear()

                # with writer.as_default():
                #     tf.summary.trace_export(name="model_trace", step=episode, profiler_outdir="./tblog")

        # END of a epoch
        save_agent_path: str = rl_config.agent_model_path() + "/epoch_{}".format(e)
        if rl_config.__REDUCE_LOADED_DATA__ is True:
            save_agent_path = save_agent_path + "_reduced"
        agent.model.save(filepath=save_agent_path, overwrite=True, include_optimizer=True)
        print("Epoch save: Agent model saved to", save_agent_path)

    return


def main():
    global __EVALUATION_MODE__, __epoch_shuffle_seeds_idx__
    parser = ArgumentParser()
    parser.add_argument("-e", "--evaluate", help="evaluate model", default=False, action='store_true')
    parser.add_argument("-v", "--verbose", help="output logs", default=False, action='store_true')

    parser.add_argument("-d", "--debug", help="debug program", default=False, action='store_true')
    parser.add_argument("-sc", "--share_classifier",
                        help="Share the classifier instance. If False, each SAEnv would have its own classifier model",
                        default=False, action='store_true')
    parser.add_argument("-r", "--reduced", help="reduced the loaded number of data", type=float, default=1.)
    parser.add_argument("-s", "--save_every_s_episode",  # now unused
                        help="specify storing frequency",
                        type=int, default=1000)

    args = vars(parser.parse_args())

    if args["evaluate"] is True:
        # TODO: implement this for RL
        print("In evaluation mode.")
        __EVALUATION_MODE__ = True
    if args["reduced"] > 1.:
        print("Reduced data to be loaded by {}.".format(args["reduced"]))
        rl_config.reduce_loaded_data(args["reduced"])

    global __OUTPUT_LOGS__
    __OUTPUT_LOGS__ = args["verbose"]

    global shared_classifier
    if args["share_classifier"] is True:
        shared_classifier = keras.models.load_model(selected_classifier_model_path)
    else:
        shared_classifier = None

    save_every_s_episode = args["save_every_s_episode"]
    if save_every_s_episode > 0 and save_every_s_episode != 1000:
        print("Save abs prcs and agent model every {} episodes".format(save_every_s_episode))

    agent = Agent()
    dataset.load(shuffle_each_category=False, verbose=True,
                      num_per_category=rl_config.num_per_category(),
                      num_training_per_category=rl_config.num_training_per_category(),
                      load_cat_only=args["debug"])
    train_multithreading(agent=agent, save_every_s_episode=save_every_s_episode)
    executor.shutdown()
    return


if __name__ == "__main__":
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    main()
