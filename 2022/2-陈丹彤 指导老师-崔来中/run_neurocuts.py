#!/usr/bin/env python

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import argparse
import os
import glob
import json

from gym.spaces import Tuple, Box, Discrete, Dict
import numpy as np

from ray.rllib.models import ModelCatalog
import ray
from ray import tune
from ray.tune import run_experiments, grid_search
from ray.tune.registry import register_env
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.evaluation.postprocessing import Postprocessing

from neurocuts_env import NeuroCutsEnv
from mask import PartitionMaskModel

parser = argparse.ArgumentParser()

#用于在命令行中选择要运行的规则
parser.add_argument("--rules",
    type=lambda expr: [
        os.path.abspath("classbench/{}".format(r)) for r in expr.split(",")],
    default="acl5_1k",
    help="Rules file name or list of rules files separated by comma.")

#将有效的树转储到此目录以供以后检查
parser.add_argument(
    "--dump-dir",
    type=str,
    default="/tmp/neurocuts_out",
    help="Dump valid trees to this directory for later inspection.")

#在开发中使用快速超参数配置进行测试
parser.add_argument(
    "--fast",
    action="store_true",
    help="Use fast hyperparam configuration for testing in development.")

#设置分区器
parser.add_argument(
    "--partition-mode",
    type=str,
    default=None,
    help="Set the partitioner: [None, 'simple', 'efficuts', 'cutsplit'].")

#用于组合深度和大小权重的函数
parser.add_argument(
    "--reward-shape",
    type=str,
    default="linear",
    help="Function to use for combining depth and size weights.")

#设置泛化估计优势中的λ，γ在环境初始化的时候默认是1
parser.add_argument("--gae-lambda", type=float, default=0.95)

#用于组合深度和大小的权重，范围是[0，1]
parser.add_argument(
    "--depth-weight",
    type=float,
    default=1.0,
    help="Weight to use for combining depth and size, in [0, 1]")

#要从RLlib请求的并行工作线程数
parser.add_argument(
    "--num-workers",
    type=int,
    default=0,
    help="Number of parallel workers to request from RLlib.")

#是否使用gpu
parser.add_argument(
    "--gpu", action="store_true", help="Whether to tell RLlib to use a GPU.")

#要连接到的现有Ray群集的地址。
parser.add_argument(
    "--redis-address",
    type=str,
    default=None,
    help="Address of existing Ray cluster to connect to.")

#on_episode_end，rllib的回调函数，当一个episode结束后调用，用于计算和记录那个episode的统计数据
#info来自neurocuts_env.py中NeuroCutsEnv类的step()函数
def on_episode_end(info):
    """Report tree custom metrics."""

    episode = info["episode"]
    info = episode.last_info_for(0)
    if not info:
        info = episode.last_info_for((0, 0))
    pid = info["rules_file"].split("/")[-1]
    out = os.path.abspath("valid_trees-{}.txt".format(pid))
    if info["nodes_remaining"] == 0:
        info["tree_depth_valid"] = info["tree_depth"]
        info["num_nodes_valid"] = info["num_nodes"]
        info["num_splits_valid"] = info["num_splits"]
        info["bytes_per_rule_valid"] = info["bytes_per_rule"]
        info["memory_access_valid"] = info["memory_access"]
        with open(out, "a") as f:
            f.write(json.dumps(info))
            f.write("\n")
    else:
        info["tree_depth_valid"] = float("nan")
        info["num_nodes_valid"] = float("nan")
        info["num_splits_valid"] = float("nan")
        info["bytes_per_rule_valid"] = float("nan")
        info["memory_access_valid"] = float("nan")
    del info["rules_file"]
    del info["tree_stats"]
    del info["tree_stats_str"]
    episode.custom_metrics.update(info)


def postprocess_gae(info):
    traj = info["post_batch"]
    infos = traj[SampleBatch.INFOS]
    traj[Postprocessing.ADVANTAGES] = np.array(
        [i["__advantage__"] for i in infos])
    traj[Postprocessing.VALUE_TARGETS] = np.array(
        [i["__value_target__"] for i in infos])
#    print("override adv and v targets", traj[Postprocessing.ADVANTAGES],
#          traj[Postprocessing.VALUE_TARGETS])


if __name__ == "__main__":
    args = parser.parse_args()
    ray.init(redis_address=args.redis_address)
    #ray.tune.register_env(name, env_creator)注册一个自定义环境。
    #name (str) –注册的名称。env_creator (obj) –创建环境的函数。
    register_env(
        "tree_env", lambda env_config: NeuroCutsEnv(
            env_config["rules"],
            max_depth=env_config["max_depth"],
            max_actions_per_episode=env_config["max_actions"],
            dump_dir=env_config["dump_dir"],
            depth_weight=env_config["depth_weight"],
            reward_shape=env_config["reward_shape"],
            partition_mode=env_config["partition_mode"],
            zero_obs=env_config["zero_obs"],
            tree_gae=env_config["tree_gae"],
            tree_gae_gamma=env_config["tree_gae_gamma"],
            tree_gae_lambda=env_config["tree_gae_lambda"]))

    ModelCatalog.register_custom_model("mask", PartitionMaskModel)

    run_experiments({
        "neurocuts_{}".format(args.partition_mode): {
            "run": "PPO",
            "env": "tree_env",
            "stop": {
                "timesteps_total": 100000 if args.fast else 10000000,
            },
            "config": {
                "log_level": "WARN",
                "num_gpus": 0.2 if args.gpu else 0,
                "num_workers": args.num_workers,
                "sgd_minibatch_size": 100 if args.fast else 1000,
                "sample_batch_size": 200 if args.fast else 5000,
                "train_batch_size": 1000 if args.fast else 15000,
                "batch_mode": "complete_episodes",
                "observation_filter": "NoFilter",
                #设置模型参数，custom_model即设置自定义模型的名字
                "model": {
                    "custom_model": "mask",
                    "fcnet_hiddens": [512, 512], #fcnet_hiddens为全连接网络隐层神经元个数列表
                },
                "vf_share_layers": False,
                "entropy_coeff": 0.01,
                "callbacks": {
                    "on_episode_end": tune.function(on_episode_end),
#                    "on_postprocess_traj": tune.function(postprocess_gae),
                },
                #grid_search()网格搜索是一种穷举方法。给定一系列超参，然后再所有超参组合中穷举遍历，
                #从所有组合中选出最优的一组超参数，其实就是暴力方法在全部解中找最优解。
                "env_config": {
                    "tree_gae": False,
                    "tree_gae_gamma": 1.0,
                    "tree_gae_lambda": grid_search([args.gae_lambda]),
                    "zero_obs": False,
                    "dump_dir": args.dump_dir,
                    "partition_mode": args.partition_mode,
                    "reward_shape": args.reward_shape,
                    "max_depth": 100 if args.fast else 500,
                    "max_actions": 1000 if args.fast else 15000,
                    "depth_weight": args.depth_weight,
                    "rules": grid_search(args.rules),
                },
            },
        },
    })
