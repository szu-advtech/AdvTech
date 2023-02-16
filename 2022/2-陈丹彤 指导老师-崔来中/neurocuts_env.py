import collections
import time
import os
import numpy as np
import pickle
from gym.spaces import Tuple, Box, Discrete, Dict
#Ray是一个用于构建和运行分布式应用程序的框架。Rllib是一个用于强化学习的开源库
#最底层的分布式计算任务由Ray引擎支撑，倒数第二层是对特定强化学习任务的抽象
#第二层是Rllib算法和开发者可以自定义的算法，最顶层是Rllib对一些应用的支持
#策略（Policies）是Rllib中的核心概念。policies是定义agent如何在环境中工作的Python类。
#在gym.env中只有一个agent和policy。在VectorEnv中，一个策略针对多个代理。在MultiAgentEnv中，可以有多个侧率多个代理。
#Rllib模块从environment开始，输入action，输出observation，obs通过预处理器preprocessor和过滤器filter，然后送入policy的model中，model的输出送入action distribution模块决定下一个动作
#其中environment、preprocessor、model和action distribution可以用户自定义
#对于preprocessor，rllib会从env的obs空间类型来选择内置预处理器，离散obs会被one-hot编码，atari obs会被下采样，tuple和dict obs会被flatten
from ray.rllib.env import MultiAgentEnv
from ray.rllib.evaluation.rollout_worker import get_global_worker

from tree import Tree, load_rules_from_file
from hicuts import HiCuts

NUM_DIMENSIONS = 5
NUM_PART_LEVELS = 6  # 2%, 4%, 8%, 16%, 32%, 64%


class NeuroCutsEnv(MultiAgentEnv):
    """NeuroCuts multi-agent tree building environment. 
    #NeuroCuts多代理树构建环境

    In this env, each "cut" in the tree is an action taken by a
    different agent. All the agents share the same policy. We
    aggregate rewards at the end of the episode and assign each
    cut its reward based on the policy performance (actual depth).
    在此环境中，树中的每个“切割”都是由不同的代理执行的操作。所有代理共享相同的策略。
    我们在每集结束时汇总奖励，并根据策略性能（实际深度）分配每个“切割”的奖励。
    
    NeuroCutsEnv继承MultiAgentEnv
    基本框架要按照MultiAgentEnv设置：
    class NeuroCutsEnv(MultiAgentEnv):
        def __init__(self,env_config):
            self.action_space=<Space>
            self.observation_space=<Space>
        def reset(self):
            return <obs>
        def step(self,action):
            return <obs>,<reward:float>,<done:bool>,<info:dict>
    """

    def __init__(self,
                 rules_file,
                 leaf_threshold=16,
                 max_cuts_per_dimension=5,
                 max_actions_per_episode=5000,
                 max_depth=100,
                 partition_mode=None, #由用户在run_neurocuts.py的时候给出
                 reward_shape="linear",
                 depth_weight=1.0,
                 dump_dir=None,
                 #GAE(eneralized advantage estimation)泛化优势估计，将λ-return用于估计优势函数，可以平衡优势函数估计中的偏差和方差
                 #一部分策略梯度（Policy Gradient）经常会选择优势函数来构造策略梯度
                 #GAE由参数γ∈[0,1]λ∈[0,1]来表示
                 tree_gae=True,
                 tree_gae_gamma=1.0,
                 tree_gae_lambda=0.95,
                 zero_obs=False):

        self.tree_gae = tree_gae
        self.tree_gae_gamma = tree_gae_gamma
        self.tree_gae_lambda = tree_gae_lambda
        self.reward_shape = {
            "linear": lambda x: x,
            "log": lambda x: np.log(x), #log()函数计算x的自然对数，即以e为底的对数
        }[reward_shape]
        self.zero_obs = zero_obs
        
        #assert用于判断一个表达式，在表达式条件为 false 的时候触发异常
        
        assert partition_mode in [None, "simple", "efficuts", "cutsplit"]
        self.partition_enabled = partition_mode == "simple" #如果partition_mode是simple，就令partion_enabled为true
        if partition_mode in ["efficuts", "cutsplit"]:
            self.force_partition = partition_mode
        else:
            self.force_partition = False

        #os.path.expanduser()函数可以将参数中的开头部分的~或~user替换为当前用户的home目录并返回。
        #如Windows下os.path.expanduser(“~/.config/”)可以得到C:\\users\\xxx/.config/
        self.dump_dir = dump_dir and os.path.expanduser(dump_dir) 
        if self.dump_dir:
            try:
                os.makedirs(self.dump_dir) #os.makedirs用于递归创建目录
            except:
                pass
        self.best_time = float("inf") #float("inf")表示正无穷
        self.best_space = float("inf")

        self.depth_weight = depth_weight
        self.rules_file = rules_file
        #load_rules_from_file是Tree.py中的函数，参数是规则文件的名字，返回一个列表，列表元素是Rule对象
        self.rules = load_rules_from_file(rules_file)
        self.leaf_threshold = leaf_threshold
        self.max_actions_per_episode = max_actions_per_episode
        self.max_depth = max_depth
        self.num_actions = None
        self.tree = None
        self.node_map = None
        self.child_map = None
        self.max_cuts_per_dimension = max_cuts_per_dimension
        if self.partition_enabled: #上面有如果partition_mode是simple，就令partion_enabled为true
            self.num_part_levels = NUM_PART_LEVELS
        else:
            self.num_part_levels = 0
        
        #每个环境都定义了自己的状态空间observation_space和动作空间action_space
        #Discrete表示离散空间，Box表示连续空间
        #Tuple用于合并两个space实例
        #Dict用于合并两个space实例
        self.action_space = Tuple([
            Discrete(NUM_DIMENSIONS),
            Discrete(max_cuts_per_dimension + self.num_part_levels)
        ])
        self.observation_space = Dict({
            #obs即observation
            "real_obs": Box(0, 99999999, (279, ), dtype=np.float32), #Box(low, high,shape,dtype)
            "action_mask": Box(
                0,
                1, (NUM_DIMENSIONS + max_cuts_per_dimension + self.num_part_levels, ),
                dtype=np.float32),
        })

    #reset()函数为初始化函数。在强化学习算法中，agent要一次次尝试，累积经验然后从经验中学到好的动作。
    #一次尝试被称为一条轨迹或一个episode。每次尝试都要到达终止状态，一次尝试结束后，agent要从头开始，这就需要agent有重新初始化的功能
    def reset(self):
        self.num_actions = 0
        self.exceeded_max_depth = []
        self.tree = Tree(
            self.rules,
            self.leaf_threshold,
            refinements={
                "node_merging": True,
                "rule_overlay": True,
                "region_compaction": False,
                "rule_pushup": False,
                "equi_dense": False,
            })
        #node_map中，key是Node对象的id，value是Node对象本身
        self.node_map = {
            self.tree.root.id: self.tree.root,
        }
        #child_map中，key是父亲Node的id，value是它所有孩子的id的列表
        self.child_map = {}

        if self.force_partition:
            if self.force_partition == "cutsplit":
                self.tree.partition_cutsplit() #tree.py中的函数，调用cutsplit分区器
            elif self.force_partition == "efficuts":
                self.tree.partition_efficuts() #tree.py中的函数，调用efficuts分区器
            else:
                assert False, self.force_partition
            for c in self.tree.root.children:
                self.node_map[c.id] = c
            self.child_map[self.tree.root.id] = [
                c.id for c in self.tree.root.children
            ]

        start = self.tree.current_node
        return {start.id: self._encode_state(start)}
    
    #step()函数在仿真器中扮演物理引擎的角色，输入是动作action，输出是<下一步状态 回报reward 是否终止done 调试项info>
    #该函数描述了agent与环境交互的所有信息，是环境文件中最重要的函数。在该函数中，一般利用agent的运动学模型和动力学模型计算下一步状态和reward，并判断是否到达终止状态
    def step(self, action_dict):
        assert len(action_dict) == 1  # one at a time processing

        new_children = []
        for node_id, action in action_dict.items():
            node = self.node_map[node_id]
            orig_action = action
            if np.isscalar(action): #isscalar()函数，判断参数是否是标量，即一行一列的矩阵
                assert not self.partition_enabled, action
                partition = False
                cut_dimension = int(action) % 5
                cut_num = int(action) // 5 #//为取整除，返回结果的整数部分
                action = [cut_dimension, cut_num]
            else:
                if action[1] >= self.max_cuts_per_dimension:
                    assert self.partition_enabled, (
                        action, self.max_cuts_per_dimension)
                    partition = True
                    action[1] -= self.max_cuts_per_dimension
                else:
                    partition = False
            
            #如果partition是true，则对树进行分区操作；否则进行切割操作
            if partition:
                children = self.tree.partition_node(node, action[0], action[1])
            else:
                cut_dimension, cut_num = self.action_tuple_to_cut(node, action)
                children = self.tree.cut_node(node, cut_dimension,
                                              int(cut_num))

            self.num_actions += 1
            num_leaf = 0
            for c in children:
                self.node_map[c.id] = c
                if not self.tree.is_leaf(c):
                    new_children.append(c)
                else:
                    num_leaf += 1
            self.child_map[node_id] = [c.id for c in children]

        node = self.tree.get_current_node()
        while node and (self.tree.is_leaf(node)
                        or node.depth > self.max_depth):
            node = self.tree.get_next_node()
            if node and node.depth > self.max_depth:
                self.exceeded_max_depth.append(node)
        nodes_remaining = self.tree.nodes_to_cut + self.exceeded_max_depth

        obs, rew, done, info = {}, {}, {}, {}

        if (not nodes_remaining
                or self.num_actions > self.max_actions_per_episode
                or self.tree.get_current_node() is None):
            zero_state = self._zeros()
            rew = self.compute_rewards(self.depth_weight) #计算reward
            stats = {}
            obs = {node_id: zero_state for node_id in rew.keys()}
            if self.tree_gae:
                advantages, stats = self.compute_gae(self.depth_weight)
                info = {
                    node_id: {
                        "__advantage__": advantages[node_id],
                        "__value_target__": rew[node_id],
                    }
                    for node_id in rew.keys()
                }
            else:
                info = {node_id: {} for node_id in rew.keys()}
            result = self.tree.compute_result()
            rules_remaining = set()
            for n in nodes_remaining:
                for r in n.rules:
                    rules_remaining.add(str(r))
            info[self.tree.root.id].update({
                "bytes_per_rule": result["bytes_per_rule"],
                "memory_access": result["memory_access"],
                "exceeded_max_depth": len(self.exceeded_max_depth),
                "tree_depth": self.tree.get_depth(),
                "tree_stats": self.tree.get_stats(),
                "tree_stats_str": self.tree.stats_str(),
                "nodes_remaining": len(nodes_remaining),
                "rules_remaining": len(rules_remaining),
                "num_nodes": len(self.node_map),
                "partition_fraction": float(
                    len([
                        n for n in self.node_map.values() if n.is_partition()
                    ])) / len(self.node_map),
                "num_splits": self.num_actions,
                "rules_file": self.rules_file,
            })
            info[self.tree.root.id].update(stats)
            if not nodes_remaining and self.dump_dir:
                self.save_if_best(result)
            return obs, rew, {"__all__": True}, info

        needs_split = [self.tree.get_current_node()]
        obs.update({s.id: self._encode_state(s) for s in needs_split})
        rew.update({s.id: 0 for s in needs_split})
        done.update({"__all__": False})
        info.update({s.id: {} for s in needs_split})
        return obs, rew, done, info
    
    #将最优的（？）树保存进pkl文件内，需要时可以pickle.load
    def save_if_best(self, result):
        time_stat = int(result["memory_access"])  #树的内存访问次数（？）
        space_stat = int(result["bytes_per_rule"]) #树的内存占用空间
        save = False
        if time_stat < self.best_time:
            self.best_time = time_stat
            save = True
        if space_stat < self.best_space:
            self.best_space = space_stat
            save = True
        if save:
            #os.path.join()用于拼接文件路径
            #pkl文件：python中有一种存储方式，可以存储为.pkl文件
            #这个存储方式可以将python项目过程中用到的一些暂时变量或者需要提取或暂存的字符串、列表等等数据保存起来
            out = os.path.join(
                self.dump_dir, "{}-{}-acc-{}-bytes-{}.pkl".format(
                    os.path.basename(self.rules_file), time_stat, space_stat,
                    time.time()))
            print("Saving tree to {}".format(out))
            #pickle.dump(obj,file,protocol=None)封装是一个将python数据对象转化为字节流的过程
            #obj为要封装的对象，file是obj要写入的文件对象，必须以二进制可写模式打开
            #将训练好的树保存到pkl文件内
            with open(out, "wb") as f:
                pickle.dump(self.tree, f)

    def action_tuple_to_cut(self, node, action):
        cut_dimension = action[0]
        range_left = node.ranges[cut_dimension * 2]
        range_right = node.ranges[cut_dimension * 2 + 1]
        cut_num = max(2, min(2**(action[1] + 1), range_right - range_left))
        return (cut_dimension, cut_num)
    
    #计算GAE
    def compute_gae(self, depth_weight):
        """Compute GAE for a branching decision environment.

           V(d) = min over nodes n at depth=d V(n)
        """

        assert depth_weight == 1.0, "GAE not supported with space weight"

        # First precompute the value of each node
        V = {}
        stats = {}
        #rllib采用多worker设计，一般包含一个主worker即localworker，负责利用收集到的数据进行训练；多个辅worker负责数据收集。
        #每个辅worker是并行收集数据的，每个辅worker中都会分配一个model用来计算actions。允许一个worker拥有多个env，每次对多个env同时计算action，更好地利用gpu
        #get_global_worker()获取当前运行的所有worker。
        ev = get_global_worker() 
        assert ev.policy_config["use_gae"], ev.policy_config["use_gae"]
        assert ev.policy_config["lambda"] == 1.0, ev.policy_config["lambda"]
        policy = ev.get_policy()
        prep = ev.preprocessors["default_policy"]
        nlist = list(self.node_map.items())
        feed_dict = {
            policy.get_placeholder("obs"): [
                prep.transform(self._encode_state(node)) for (_, node) in nlist
            ],
            policy.get_placeholder("prev_actions"): [[0, 0] for _ in nlist],
            policy.get_placeholder("prev_rewards"): [0.0 for _ in nlist],
            policy.model.seq_lens: [1 for _ in nlist],
        }
        #sess指session会话，这条语句可能是在设置会话中的变量
        vf = policy.sess.run(policy.value_function, feed_dict)
        V_root = 0.0
        #
        for (node_id, _), v in zip(nlist, list(vf)):
            V[node_id] = v
            if node_id == self.tree.root.id:
                V_root = v

#        print(
#            "Computed node values",
#            "mean", np.mean(vf), "min", np.min(vf), "max", np.max(vf),
#            "count", len(vf))
        stats["V_gae_min"] = float(np.min(vf))
        stats["V_gae_max"] = float(np.max(vf))
        stats["V_gae_mean"] = float(np.mean(vf))
        stats["V_gae_root"] = float(V_root)

        gamma = self.tree_gae_gamma
        lambd = self.tree_gae_lambda

        # Map from node_id -> depth -> min(V at depth)
        # These values are unique per (node_id, depth) combination
        min_V_for_node = collections.defaultdict(dict)

        # Then, compute the min V at each level for each subtree
        incomplete = True
        while incomplete:
            incomplete = False
            for node_id, node in self.node_map.items():
                if node_id in min_V_for_node:
                    continue

                children = self.child_map.get(node_id, [])
                if self.tree.is_leaf(node):
                    min_V_for_node[node_id][node.depth] = -1
                elif not children:
                    min_V_for_node[node_id][node.depth] = V[node_id]
                elif all((c_id in min_V_for_node) for c_id in children):
                    min_V = {}
                    for c_id in children:
                        for depth, minv in min_V_for_node[c_id].items():
                            if depth not in min_V:
                                min_V[depth] = minv
                            else:
                                min_V[depth] = min(min_V[depth], minv)
                    assert node.depth not in min_V, min_V
                    min_V[node.depth] = V[node_id]
                    min_V_for_node[node_id] = min_V
                else:
                    incomplete = True
                    continue
#                print(
#                    "Computed minV for node", node_id, "depth", node.depth,
#                    min_V_for_node[node_id])

# delta(V)_{t+1} in the GAE paper

        def deltaV(node_id, depth):
            dv = -1 + gamma * min_V_for_node[node_id].get(d + 1, 0.0)
            dv -= min_V_for_node[node_id][d]
            return dv

        # Now we can compute GAE estimates for each
        advantages = {}
        adv_list = []
        adv_root = 0.0
        for node_id, node in self.node_map.items():
            A_gae = 0.0
            d = node.depth
            while d in min_V_for_node[node_id]:
                A_gae += (gamma * lambd)**(d - node.depth) * deltaV(node_id, d)
                d += 1
#            print("A_gae for node", node_id, "depth", node.depth, A_gae)
            adv_list.append(A_gae)
            if node_id == self.tree.root.id:
                adv_root = A_gae
            advantages[node_id] = A_gae


#        print(
#            "GAE advantages",
#            "min", np.min(adv_list), "max", np.max(adv_list),
#            "mean", np.mean(adv_list))
        stats["A_gae_min"] = float(np.min(adv_list))
        stats["A_gae_max"] = float(np.max(adv_list))
        stats["A_gae_mean"] = float(np.mean(adv_list))
        stats["A_gae_root"] = float(adv_root)

        return advantages, stats
    
    #Reward 定义了强化学习问题中的目标。
    #在每个时间步，环境向agent发送一个称为reward的单个数字。
    #Agent的唯一目标是最大化其长期收到的total reward。因此，reward定义了对于agent什么是好的什么是坏的。
    #Reward 是改变policy的主要依据;如果policy选择的action之后得到的是低奖励，
    #则可以更改policy以在将来选择该情况下的某些其他action。
    #该RL的reward是节点深度和权重的函数。目的就是使树的深度尽量小，以达到优化分类时间和/或内存占用的目的
    def compute_rewards(self, depth_weight):
        depth_to_go = collections.defaultdict(int)
        nodes_to_go = collections.defaultdict(int)
        num_updates = 1
        while num_updates > 0:
            num_updates = 0
            for node_id, node in self.node_map.items():
                if node_id not in depth_to_go:
                    if self.tree.is_leaf(node):
                        depth_to_go[node_id] = 0
                        nodes_to_go[node_id] = 0
                    else:
                        depth_to_go[node_id] = 1
                        nodes_to_go[node_id] = 1
                if node_id in self.child_map:
                    if self.node_map[node_id].is_partition():
                        max_child_depth = self.tree_gae_gamma * sum(
                            [depth_to_go[c] for c in self.child_map[node_id]])
                    else:
                        max_child_depth = 1 + self.tree_gae_gamma * max(
                            [depth_to_go[c] for c in self.child_map[node_id]])
                    if max_child_depth > depth_to_go[node_id]:
                        depth_to_go[node_id] = max_child_depth
                        num_updates += 1
                    sum_child_cuts = len(self.child_map[node_id]) + sum(
                        [nodes_to_go[c] for c in self.child_map[node_id]])
                    if sum_child_cuts > nodes_to_go[node_id]:
                        nodes_to_go[node_id] = sum_child_cuts
                        num_updates += 1
        rew = {
            node_id:
            -depth_weight * self.reward_shape(depth) - (1.0 - depth_weight) *
            self.reward_shape(float(nodes_to_go[node_id]))
            for (node_id, depth) in depth_to_go.items()
            if node_id in self.child_map
        }
        return rew

    def _zeros(self):
        zeros = np.array([0] * 279)
        return {
            "real_obs": zeros,
            "action_mask": np.array([1] *
                (5 + self.max_cuts_per_dimension + self.num_part_levels)),
        }

    def _encode_state(self, node):
        if node.depth > 1:
            #action_mask的前(NUM_DIMENSIONS + self.max_cuts_per_dimension)个元素是1，后self.num_part_levels个元素是0
            action_mask = ([1] * (NUM_DIMENSIONS + self.max_cuts_per_dimension) +
                           [0] * self.num_part_levels)
        else:
            assert node.depth == 1, node.depth
            action_mask = ([1] * (NUM_DIMENSIONS + self.max_cuts_per_dimension)
                           + [1] * self.num_part_levels)
        #tree.py里Node类的函数，获得Node对象的state包括ranges和分区的状态
        s = np.array(node.get_state())
        return {
            #zeros_like(a)函数是构建一个与a同维度的数组，并初始化所有变量为0
            "real_obs": np.zeros_like(s) if self.zero_obs else s,
            "action_mask": np.array(action_mask),
        }
