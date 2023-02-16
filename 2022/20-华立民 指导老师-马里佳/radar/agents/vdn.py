from radar.agents.dqn import DQNLearner
import torch
import torch.nn.functional as F

class VDNLearner(DQNLearner):

    def __init__(self, params):
        self.global_value_network = None
        self.global_target_network = None
        super(VDNLearner, self).__init__(params)
        self.zero_actions = torch.zeros(self.batch_size, dtype=torch.long).unsqueeze(1)

    def update(self, state, obs, joint_action, rewards, next_state, next_obs, dones, is_adversary):
        super(VDNLearner, self).update_transition(state, obs, joint_action, rewards, next_state, next_obs, dones, is_adversary)
        self.warmup_phase = max(0, self.warmup_phase-1)
        if self.warmup_phase <= 0:
            self.training_count += 1
            self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
            minibatch = self.memory.sample_batch(self.batch_size)
            nr_adversaries = len(self.adversary_ids)
            nr_protagonists = self.nr_agents - nr_adversaries
            if not is_adversary:
                self.train_step(minibatch, False, nr_protagonists)
            elif self.adversary_ratio > 0:
                self.train_step(minibatch, True, nr_adversaries)
            self.update_target_network()
            return True
        return False

    def global_value(self, network, Q_values, states):
        return Q_values.sum(1)
        
    def train_step(self, minibatch, is_adversary, nr_agents):
        minibatch_data = self.collect_minibatch_data(minibatch)
        return self.train_step_with(minibatch_data, is_adversary, nr_agents=nr_agents)

    def train_step_with(self, minibatch_data, is_adversary, target_values=None, nr_agents=None):
        states = minibatch_data["states"]
        next_states = minibatch_data["next_states"]
        if is_adversary:
            histories = minibatch_data["adv_histories"]
            actions = minibatch_data["adv_actions"]
            next_histories = minibatch_data["next_adv_histories"]
            rewards = minibatch_data["adv_rewards"]
            optimizer = self.adversary_optimizer
        else:
            histories = minibatch_data["pro_histories"]
            actions = minibatch_data["pro_actions"]
            next_histories = minibatch_data["next_pro_histories"]
            rewards = minibatch_data["pro_rewards"]
            optimizer = self.protagonist_optimizer
        if nr_agents is None:
            nr_agents = self.nr_agents
        state_action_values = self.policy_net(histories, is_adversary).gather(1, actions.unsqueeze(1)).squeeze()
        state_action_values = state_action_values.view(-1, nr_agents)
        state_action_values = self.global_value(self.global_value_network, state_action_values, states)
        if target_values is None:
            rewards = rewards.view(-1, nr_agents)
            assert rewards.size(0) == states.size(0)
            rewards = rewards.gather(1, self.zero_actions).squeeze()
            next_state_values = self.target_net(next_histories, is_adversary).max(1)[0]
            next_state_values = next_state_values.view(-1, nr_agents)
            next_state_values = self.global_value(self.global_target_network, next_state_values, next_states).detach()
            target_values = rewards + self.gamma*next_state_values
        optimizer.zero_grad()
        loss = F.mse_loss(state_action_values, target_values)
        loss.backward()
        optimizer.step()
        return True