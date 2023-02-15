import random
import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from radar.agents.controller import DeepLearningController, get_param_or_default
from radar.modules import MLP, AdversarialModule

class PPONet(nn.Module):
    def __init__(self, input_shape, nr_actions, max_history_length, q_values=False):
        super(PPONet, self).__init__()
        self.fc_net = MLP(input_shape, max_history_length)
        self.action_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
        if q_values:
            self.value_head = nn.Linear(self.fc_net.nr_hidden_units, nr_actions)
        else:
            self.value_head = nn.Linear(self.fc_net.nr_hidden_units, 1)

    def forward(self, x, use_gumbel_softmax=False):
        x = self.fc_net(x)
        if use_gumbel_softmax:
            return F.gumbel_softmax(self.action_head(x), hard=True, dim=-1), self.value_head(x)
        return F.softmax(self.action_head(x), dim=-1), self.value_head(x)

class PPOLearner(DeepLearningController):

    def __init__(self, params):
        super(PPOLearner, self).__init__(params)
        self.nr_epochs = get_param_or_default(params, "nr_epochs", 5)
        self.nr_episodes = get_param_or_default(params, "nr_episodes", 10)
        self.eps_clipping = get_param_or_default(params, "eps_clipping", 0.2)
        self.use_q_values = get_param_or_default(params, "use_q_values", False)
        history_length = self.max_history_length
        input_shape = self.input_shape
        nr_actions = self.nr_actions
        network_constructor = lambda in_shape, actions, length: PPONet(in_shape, actions, length, self.use_q_values)
        self.policy_net = AdversarialModule(input_shape, nr_actions, history_length, network_constructor).to(self.device)
        self.protagonist_optimizer = torch.optim.Adam(self.policy_net.protagonist_parameters(), lr=self.alpha)
        self.adversary_optimizer = torch.optim.Adam(self.policy_net.adversary_parameters(), lr=self.alpha)

    def joint_action_probs(self, histories, training_mode=True, agent_ids=None):
        action_probs = []
        if agent_ids is None:
            agent_ids = self.agent_ids
        for i, agent_id in enumerate(agent_ids):
            history = [[joint_obs[i]] for joint_obs in histories]
            history = torch.tensor(history, device=self.device, dtype=torch.float32)
            is_adversary = agent_id in self.adversary_ids
            probs, value = self.policy_net(history, is_adversary)
            assert len(probs) == 1, "Expected length 1, but got shape {}".format(probs.shape)
            probs = probs.detach().numpy()[0]
            value = value.detach()
            action_probs.append(probs)
        return action_probs

    def policy_update(self, minibatch_data, optimizer, is_adversary, random_agent_indices=None):
        if is_adversary:
            returns = minibatch_data["adv_returns"]
            actions = minibatch_data["adv_actions"]
            old_probs = minibatch_data["adv_action_probs"]
            histories = minibatch_data["adv_histories"]
        else:
            returns = minibatch_data["pro_returns"]
            actions = minibatch_data["pro_actions"]
            old_probs = minibatch_data["pro_action_probs"]
            histories = minibatch_data["pro_histories"]
        action_probs, expected_values = self.policy_net(histories, is_adversary)
        policy_losses = []
        value_losses = []
        for probs, action, value, R, old_prob in zip(action_probs, actions, expected_values, returns, old_probs):
            value_index = 0
            if self.use_q_values:
                value_index = action
                advantage = value[value_index].detach()
            else:
                advantage = R - value.item()
            policy_losses.append(self.policy_loss(advantage, probs, action, old_prob))
            value_losses.append(F.mse_loss(value[value_index], torch.tensor(R)))
        value_loss = torch.stack(value_losses).mean()
        policy_loss = torch.stack(policy_losses).mean()
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return True

    def policy_loss(self, advantage, probs, action, old_prob):
        m1 = Categorical(probs)
        m2 = Categorical(old_prob)
        ratio = torch.exp(m1.log_prob(action) - m2.log_prob(action))
        clipped_ratio = torch.clamp(ratio, 1-self.eps_clipping, 1+self.eps_clipping)
        surrogate_loss1 = ratio*advantage
        surrogate_loss2 = clipped_ratio*advantage
        return -torch.min(surrogate_loss1, surrogate_loss2)

    def value_update(self, minibatch_data, is_adversary):
        pass

    def update(self, state, observations, joint_action, rewards, next_state, next_observations, dones, is_adversary):
        super(PPOLearner, self).update(state, observations, joint_action, rewards, next_state, next_observations, dones, is_adversary)
        global_terminal_reached = not [d for i,d in enumerate(dones) if (not d) and (i not in self.adversary_ids)]
        if global_terminal_reached and self.memory.size() >= self.nr_episodes:
            is_protagonist = not is_adversary
            has_adversaries = self.get_nr_adversaries() > 0
            trainable_setting = is_protagonist or has_adversaries
            if trainable_setting:
                batch = self.memory.sample_batch(self.memory.capacity)
                minibatch_data = self.collect_minibatch_data(batch, whole_batch=True)
                self.value_update(minibatch_data, is_adversary)
                for _ in range(self.nr_epochs):
                    if is_adversary:
                        optimizer = self.adversary_optimizer
                    else:
                        optimizer = self.protagonist_optimizer
                    self.policy_update(minibatch_data, optimizer, is_adversary)
            self.memory.clear()
            return True
        return False