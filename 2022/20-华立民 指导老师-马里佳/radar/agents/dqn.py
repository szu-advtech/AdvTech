from radar.agents.controller import DeepLearningController, ReplayMemory
from radar.modules import MLP, AdversarialModule
from radar.utils import argmax, get_param_or_default
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy

class DQNNet(torch.nn.Module):
    def __init__(self, input_shape, outputs, max_history_length):
        super(DQNNet, self).__init__()
        self.fc_net = MLP(input_shape, max_history_length)
        self.action_head = nn.Linear(self.fc_net.nr_hidden_units, outputs)

    def forward(self, x):
        x = self.fc_net(x)
        return self.action_head(x)

class DQNLearner(DeepLearningController):

    def __init__(self, params):
        super(DQNLearner, self).__init__(params)
        self.epsilon = 1.0
        self.epsilon_decay = get_param_or_default(params, "epsilon_decay", 0.0001)
        self.epsilon_min = get_param_or_default(params, "epsilon_min", 0.01)
        self.batch_size = get_param_or_default(params, "batch_size", 64)
        history_length = self.max_history_length
        input_shape = self.input_shape
        nr_actions = self.nr_actions
        self.policy_net = AdversarialModule(input_shape, nr_actions, history_length, DQNNet).to(self.device)
        self.target_net = AdversarialModule(input_shape, nr_actions, history_length, DQNNet).to(self.device)
        self.protagonist_optimizer = torch.optim.Adam(self.policy_net.protagonist_parameters(), lr=self.alpha)
        self.adversary_optimizer = torch.optim.Adam(self.policy_net.adversary_parameters(), lr=self.alpha)
        self.update_target_network()

    def joint_action_probs(self, histories, training_mode=True, agent_ids=None):
        action_probs = []
        used_epsilon = self.epsilon_min
        if training_mode:
            used_epsilon = self.epsilon
        if agent_ids is None:
            agent_ids = self.agent_ids
        for i, agent_id in enumerate(agent_ids):
            history = [[joint_obs[i]] for joint_obs in histories]
            history = torch.tensor(history, device=self.device, dtype=torch.float32)
            is_adversary = agent_id in self.adversary_ids
            Q_values = self.policy_net(history, is_adversary).detach().numpy()
            assert len(Q_values) == 1, "Expected length 1, but got shape {}".format(Q_values.shape)
            probs = used_epsilon*numpy.ones(self.nr_actions)/self.nr_actions
            rest_prob = 1 - sum(probs)
            probs[argmax(Q_values[0])] += rest_prob
            action_probs.append(probs/sum(probs))
        return action_probs

    def update(self, state, obs, joint_action, rewards, next_state, next_obs, dones, is_adversary):
        super(DQNLearner, self).update(state, obs, joint_action, rewards, next_state, next_obs, dones, is_adversary)
        if self.warmup_phase <= 0:
            minibatch = self.memory.sample_batch(self.batch_size)
            minibatch_data = self.collect_minibatch_data(minibatch)
            if not is_adversary:
                histories = minibatch_data["pro_histories"]
                next_histories = minibatch_data["next_pro_histories"]
                actions = minibatch_data["pro_actions"]
                rewards = minibatch_data["pro_rewards"]
                self.update_step(histories, next_histories, actions, rewards, self.protagonist_optimizer, False)
            elif self.adversary_ratio > 0:
                histories = minibatch_data["adv_histories"]
                next_histories = minibatch_data["next_adv_histories"]
                actions = minibatch_data["adv_actions"]
                rewards = minibatch_data["adv_rewards"]
                self.update_step(histories, next_histories, actions, rewards, self.adversary_optimizer, True)
            self.update_target_network()
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
            self.training_count += 1
            return True
        return False

    def update_step(self, histories, next_histories, actions, rewards, optimizer, is_adversary):
        Q_values = self.policy_net(histories, is_adversary).gather(1, actions.unsqueeze(1)).squeeze()
        next_Q_values = self.target_net(next_histories, is_adversary).max(1)[0].detach()
        target_Q_values = rewards + self.gamma*next_Q_values
        optimizer.zero_grad()
        loss = F.mse_loss(Q_values, target_Q_values)
        loss.backward()
        optimizer.step()
        return loss
