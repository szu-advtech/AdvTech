import random
import numpy
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from radar.utils import get_param_or_default
from radar.agents.ppo import PPOLearner

class PPOMIXLearner(PPOLearner):

    def __init__(self, params):
        self.global_input_shape = params["global_observation_shape"]
        super(PPOMIXLearner, self).__init__(params)
        self.central_q_learner = params["central_q_learner"]
        self.last_q_loss = 0

    def value_update(self, minibatch_data, is_adversary):
        batch_size = minibatch_data["states"].size(0)
        self.central_q_learner.zero_actions = torch.zeros(batch_size, dtype=torch.long).unsqueeze(1)
        nr_agents = self.get_nr_protagonists()
        if not is_adversary:
            returns = minibatch_data["pro_returns"].view(-1, nr_agents)
        else:
            returns = minibatch_data["adv_returns"].view(-1, nr_agents)
        returns = returns.gather(1, self.central_q_learner.zero_actions).squeeze()
        returns /= self.nr_agents
        returns *= nr_agents
        assert returns.size(0) == batch_size
        for _ in range(self.nr_epochs):
            self.last_q_loss = self.central_q_learner.train_step_with(minibatch_data, is_adversary, returns, nr_agents)

    def policy_update(self, minibatch_data, optimizer, is_adversary):
        if is_adversary:
            old_probs = minibatch_data["adv_action_probs"]
            histories = minibatch_data["adv_histories"]
            actions = minibatch_data["adv_actions"]
            returns = minibatch_data["adv_returns"]
        else:
            old_probs = minibatch_data["pro_action_probs"]
            histories = minibatch_data["pro_histories"]
            actions = minibatch_data["pro_actions"]
            returns = minibatch_data["pro_returns"]
        action_probs, expected_values = self.policy_net(histories, is_adversary)
        expected_Q_values = self.central_q_learner.policy_net(histories, is_adversary).detach()
        policy_losses = []
        value_losses = []
        for probs, action, value, Q_values, old_prob, R in\
            zip(action_probs, actions, expected_values, expected_Q_values, old_probs, returns):
            baseline = sum(probs*Q_values)
            baseline = baseline.detach()
            advantage = R - baseline
            policy_losses.append(self.policy_loss(advantage, probs, action, old_prob))
            value_losses.append(F.mse_loss(value[0], Q_values[action]))
        policy_loss = torch.stack(policy_losses).mean()
        value_loss = torch.stack(value_losses).mean()
        loss = policy_loss + value_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return True