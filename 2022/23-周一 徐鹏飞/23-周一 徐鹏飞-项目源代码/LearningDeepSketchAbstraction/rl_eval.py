import os
import matplotlib.pyplot as plt
from abstraction_process import read_sketch_abs_prcs


def to_rewards(log_msg: str):
    rewards = []
    idx = log_msg.find('[')
    if idx != -1:
        reward_msg = log_msg[idx + 1: len(log_msg) - 2]
        split = reward_msg.split(" ")
        for e in split:
            if e != '':
                value = float(e.replace(",", ""))
                rewards.append(value)
    # print(rewards)
    return rewards


def plot_positive_ratio(rewards):
    start = 0
    delta = 1000
    intervals = []
    ratios = []
    while start < len(rewards):
        positive_num: int = 0
        intervals.append(start)
        for i in range(start, min(start + delta, len(rewards))):
            if rewards[i] > 0.:
                positive_num += 1
        positive_ratio: float = positive_num / delta
        ratios.append(positive_ratio)
        start += delta

    plt.plot(intervals, ratios, 'b')

    plt.xlabel('sketch range')
    plt.ylabel('positive rewards ratios')
    plt.show()
    plt.clf()
    return


def plot_positive_avg_rewards(rewards):
    start = 0
    delta = 500
    intervals = []
    avg_reward = []
    while start < len(rewards):
        total_rewards: float = sum([r for r in rewards[start: start + delta]])
        avg_reward.append(total_rewards / delta)
        intervals.append(start)

        start += delta

    plt.plot(intervals, avg_reward, 'b')

    plt.xlabel('sketch range')
    plt.ylabel('avg reward')
    plt.show()
    plt.clf()
    return


def plot_rl_train_log():
    rl_log_filename: str = "./rl_train.log"
    log_file = open(rl_log_filename, "r")

    i: int = 0
    batch_size: int = 5
    rewards = []
    while True:
        log_msg: str = log_file.readline()
        if log_msg == "":
            break

        rewards.extend(to_rewards(log_msg))

    plot_positive_avg_rewards(rewards)
    return


def plot_removed_strokes_num(ep_start=-1, ep_end=-1):
    abs_dir = "./abstraction_process"
    abs_ep_dirs = os.listdir(path=abs_dir)
    abs_ep_dirs.sort(key=int)

    if 0 <= ep_start < ep_end:
        abs_ep_dirs_temp = []

        for abs_ep_dir_name in abs_ep_dirs:
            if ep_start <= int(abs_ep_dir_name) < ep_end:
                abs_ep_dirs_temp.append(abs_ep_dir_name)

        abs_ep_dirs = abs_ep_dirs_temp

    removed_stroke_num_eps = []

    for abs_ep_dir_name in abs_ep_dirs:
        print("reading episode", abs_ep_dir_name)
        removed_stroke_num_ep: int = 0

        abs_ep_dir_path = abs_dir + "/" + abs_ep_dir_name
        abs_prcs = os.listdir(path=abs_ep_dir_path)
        for abs_prc in abs_prcs:
            sketch_np, removed_stroke_idx = read_sketch_abs_prcs(abs_ep_dir_path + "/" + abs_prc)
            removed_stroke_num_ep += len(removed_stroke_idx)

        removed_stroke_num_eps.append(removed_stroke_num_ep)

    plt.plot(range(len(removed_stroke_num_eps)), removed_stroke_num_eps, 'b')
    plt.show()
    return


def main():
    plot_removed_strokes_num(0, 20)
    # plot_positive_ratio(rewards)
    # plot_rl_train_log()
    return


if __name__ == '__main__':
    main()
