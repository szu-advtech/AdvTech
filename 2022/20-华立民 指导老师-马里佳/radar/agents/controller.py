import random
import numpy
import torch
from radar.utils import get_param_or_default
from radar.utils import pad_or_truncate_sequences

class ReplayMemory:

    def __init__(self, capacity, is_prioritized=False):
        self.transitions = []
        self.capacity = capacity
        self.nr_transitions = 0

    def save(self, transition):
        self.transitions.append(transition)
        self.nr_transitions += len(transition[0])
        if self.nr_transitions > self.capacity:
            removed_transition = self.transitions.pop(0)
            self.nr_transitions -= len(removed_transition[0])

    def sample_batch(self, minibatch_size):
        nr_episodes = self.size()
        if nr_episodes > minibatch_size:
            return random.sample(self.transitions, minibatch_size)
        return self.transitions

    def clear(self):
        self.transitions.clear()
        self.nr_transitions = 0

    def size(self):
        return len(self.transitions)

class Controller:

    def __init__(self, params):
        self.nr_agents = params["nr_agents"]
        self.nr_actions = params["nr_actions"]
        self.actions = list(range(self.nr_actions))
        self.agent_ids = list(range(self.nr_agents))
        self.adversary_ratio = get_param_or_default(params, "adversary_ratio", 0)
        self.randomized_adversary_ratio = False
        if self.adversary_ratio is None:
            self.randomized_adversary_ratio = True
            self.adversary_ratio = numpy.random.rand()
        self.adversary_ids = []
        self.gamma = params["gamma"]
        self.alpha = get_param_or_default(params, "alpha", 0.001)
        self.env = params["env"]

    def get_nr_adversaries(self):
        return len(self.adversary_ids)

    def get_nr_protagonists(self):
        return self.nr_agents - self.get_nr_adversaries()

    def generate_adversary_ids(self, is_adversary):
        self.adversary_ids = []
        if self.adversary_ratio > 0:
            nr_adversaries = int(self.nr_agents*self.adversary_ratio)
            if is_adversary:
                nr_adversaries = max(nr_adversaries, 1) # Ensure at least one adversary
            nr_adversaries = min(nr_adversaries, self.nr_agents-1) # Ensure at least one protagonist
            self.adversary_ids = random.sample(self.agent_ids, k=nr_adversaries)
        return self.adversary_ids

    def sample_adversary_ratio(self):
        if self.randomized_adversary_ratio:
            self.adversary_ratio = numpy.random.rand()
        return self.adversary_ratio

    def policy(self, observations, training_mode=True):
        random_joint_action = [random.choice(self.actions) \
            for agent in self.env.agents]
        return random_joint_action

    def update(self, state, observations, joint_action, rewards, next_state, next_observations, dones, is_adversary):
        return True

class DeepLearningController(Controller):
    
    def __init__(self, params):
        super(DeepLearningController, self).__init__(params)
        self.device = torch.device("cpu")
        self.use_global_reward = get_param_or_default(params, "use_global_reward", True)
        self.input_shape = params["local_observation_shape"]
        self.global_input_shape = params["global_observation_shape"]
        self.memory = ReplayMemory(params["memory_capacity"])
        self.warmup_phase = params["warmup_phase"]
        self.episode_transitions = []                
        self.max_history_length = get_param_or_default(params, "max_history_length", 1)
        self.target_update_period = params["target_update_period"]
        self.epsilon = 1
        self.training_count = 0
        self.current_histories = []
        self.current_pro_histories = []
        self.current_adv_histories = []
        self.eps = numpy.finfo(numpy.float32).eps.item()
        self.policy_net = None
        self.target_net = None
        self.default_observations = [numpy.zeros(self.input_shape) for _ in range(self.nr_agents)]

    def save_weights(self, path):
        if self.policy_net is not None:
            self.policy_net.save_weights(path)

    def load_weights(self, path):
        if self.policy_net is not None:
            self.policy_net.load_weights(path)

    def extend_histories(self, histories, observations):
        histories = histories + [observations]
        return pad_or_truncate_sequences(histories, self.max_history_length, self.default_observations)

    def policy(self, observations, training_mode=True):
        self.current_histories = self.extend_histories(self.current_histories, observations)
        action_probs = self.joint_action_probs(self.current_histories, training_mode)
        return [numpy.random.choice(self.actions, p=probs) for probs in action_probs]

    def joint_action_probs(self, histories, training_mode=True, agent_ids=None):
        if agent_ids is None:
            agent_ids = self.agent_ids
        return [numpy.ones(self.nr_actions)/self.nr_actions for _ in agent_ids]

    def update(self, state, observations, joint_action, rewards, next_state, next_observations, dones, is_adversary):
        return self.update_transition(state, observations, joint_action, rewards, next_state, next_observations, dones, is_adversary)

    def update_transition(self, state, obs, joint_action, rewards, next_state, next_obs, dones, is_adversary):
        self.warmup_phase = max(0, self.warmup_phase-1)
        if self.use_global_reward:
            global_reward = sum(rewards)
            rewards = [global_reward for _ in range(self.nr_agents)]
        pro_obs = []
        adv_obs = []
        pro_actions = []
        adv_actions = []
        next_pro_obs = []
        next_adv_obs = []
        pro_rewards = []
        adv_rewards = []
        for i in range(self.nr_agents):
            if i in self.adversary_ids:
                adv_obs.append(obs[i])
                adv_actions.append(joint_action[i])
                next_adv_obs.append(next_obs[i])
                adv_rewards.append(-rewards[i])
            else:
                pro_obs.append(obs[i])
                pro_actions.append(joint_action[i])
                next_pro_obs.append(next_obs[i])
                pro_rewards.append(rewards[i])
        protagonist_ids = [i for i in self.agent_ids if i not in self.adversary_ids]
        self.current_pro_histories = self.extend_histories(self.current_pro_histories, pro_obs)
        self.current_adv_histories = self.extend_histories(self.current_adv_histories, adv_obs)
        pro_probs = self.joint_action_probs(self.current_pro_histories, training_mode=True, agent_ids=protagonist_ids)
        adv_probs = self.joint_action_probs(self.current_adv_histories, training_mode=True, agent_ids=self.adversary_ids)
        self.episode_transitions.append((state, pro_obs, adv_obs, pro_actions, adv_actions,\
            pro_probs, adv_probs,pro_rewards, adv_rewards, next_state, next_pro_obs, next_adv_obs))
        global_terminal_reached = not [d for i,d in enumerate(dones) if (not d) and (i not in self.adversary_ids)]
        if global_terminal_reached:
            s, pro_obs, adv_obs, a1, a2, p1, p2, pro_rewards, adv_rewards, sn, next_pro_obs, next_adv_obs = tuple(zip(*self.episode_transitions))
            R1 = self.to_returns(pro_rewards, protagonist_ids)
            R2 = self.to_returns(adv_rewards, self.adversary_ids)
            self.memory.save((s, pro_obs, adv_obs, a1, a2, p1, p2, pro_rewards, adv_rewards, sn, next_pro_obs, next_adv_obs, R1,R2))
            self.episode_transitions.clear()
            self.current_histories.clear()
            self.current_pro_histories.clear()
            self.current_adv_histories.clear()
        return True

    def collect_minibatch_data(self, minibatch, whole_batch=False):
        states = []
        pro_histories = []
        adv_histories = []
        next_states = []
        next_pro_histories = []
        next_adv_histories = []
        pro_returns = []
        adv_returns = []
        pro_action_probs = []
        adv_action_probs = []
        pro_actions = []
        adv_actions = []
        pro_rewards = []
        adv_rewards = []
        max_length = self.max_history_length
        for episode in minibatch:
            states_, pro_obs, adv_obs, p_actions, a_actions, pro_probs, adv_probs,\
                p_rewards, a_rewards, next_states_, next_pro_obs, next_adv_obs, p_R, a_R = episode
            min_index = -max_length+1
            max_index = len(pro_obs)-max_length
            if whole_batch:
                indices = range(min_index, max_index)
            else:
                indices = [numpy.random.randint(min_index, max_index)]
            for index_ in indices:
                end_index = index_+max_length
                index = max(0, index_)
                assert index < end_index
                history = pad_or_truncate_sequences(list(pro_obs[index:index+max_length]), max_length, self.default_observations)
                pro_histories.append(history)
                next_history = pad_or_truncate_sequences(list(next_pro_obs[index:index+max_length]), max_length, self.default_observations)
                next_pro_histories.append(next_history)
                history = pad_or_truncate_sequences(list(adv_obs[index:index+max_length]), max_length, self.default_observations)
                adv_histories.append(history)
                next_history = pad_or_truncate_sequences(list(next_adv_obs[index:index+max_length]), max_length, self.default_observations)
                next_adv_histories.append(next_history)
                states.append(states_[end_index-1])
                next_states.append(next_states_[end_index-1])
                pro_action_probs += list(pro_probs[end_index-1])
                adv_action_probs += list(adv_probs[end_index-1])
                pro_actions += list(p_actions[end_index-1])
                adv_actions += list(a_actions[end_index-1])
                pro_rewards += list(p_rewards[end_index-1])
                pro_returns += list(p_R[end_index-1])
                adv_rewards += list(a_rewards[end_index-1])
                adv_returns += list(a_R[end_index-1])
        pro_histories = self.reshape_histories(pro_histories)
        next_pro_histories = self.reshape_histories(next_pro_histories)
        adv_histories = self.reshape_histories(adv_histories)
        next_adv_histories = self.reshape_histories(next_adv_histories)
        pro_returns = self.normalized_returns(numpy.array(pro_returns))
        adv_returns = self.normalized_returns(numpy.array(adv_returns))
        return {"states":torch.tensor(states, device=self.device, dtype=torch.float32),\
            "pro_histories":torch.tensor(pro_histories, device=self.device, dtype=torch.float32),\
            "adv_histories":torch.tensor(adv_histories, device=self.device, dtype=torch.float32),\
            "pro_actions":torch.tensor(pro_actions, device=self.device, dtype=torch.long),\
            "adv_actions":torch.tensor(adv_actions, device=self.device, dtype=torch.long),\
            "pro_action_probs":torch.tensor(pro_action_probs, device=self.device, dtype=torch.float32),\
            "adv_action_probs":torch.tensor(adv_action_probs, device=self.device, dtype=torch.float32),\
            "pro_rewards":torch.tensor(pro_rewards, device=self.device, dtype=torch.float32),\
            "adv_rewards":torch.tensor(adv_rewards, device=self.device, dtype=torch.float32),\
            "next_states":torch.tensor(next_states, device=self.device, dtype=torch.float32),\
            "next_pro_histories":torch.tensor(next_pro_histories, device=self.device, dtype=torch.float32),\
            "next_adv_histories":torch.tensor(next_adv_histories, device=self.device, dtype=torch.float32),\
            "pro_returns":torch.tensor(pro_returns, device=self.device, dtype=torch.float32),\
            "adv_returns":torch.tensor(adv_returns, device=self.device, dtype=torch.float32)}

    def reshape_histories(self, history_batch):
        histories = []
        for i in range(self.max_history_length):
            joint_observations = []
            for joint_history in history_batch:
                joint_observations += joint_history[i]
            histories.append(joint_observations)
        return histories

    def to_returns(self, individual_rewards, agent_ids):
        R = numpy.zeros(len(agent_ids))
        discounted_returns = []
        for rewards in reversed(individual_rewards):
            R = numpy.array(rewards) + self.gamma*R
            discounted_returns.append(R)
        discounted_returns.reverse()
        return numpy.array(discounted_returns)

    def normalized_returns(self, discounted_returns):
        R_mean = numpy.mean(discounted_returns)
        R_std = numpy.std(discounted_returns)
        return (discounted_returns - R_mean)/(R_std + self.eps)

    def update_target_network(self):
        target_net_available = self.target_net is not None
        if target_net_available and self.training_count % self.target_update_period is 0:
            self.target_net.protagonist_net.load_state_dict(self.policy_net.protagonist_net.state_dict())
            self.target_net.protagonist_net.eval()
            self.target_net.adversary_net.load_state_dict(self.policy_net.adversary_net.state_dict())
            self.target_net.adversary_net.eval()
