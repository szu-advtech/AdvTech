from radar.agents.a2c import A2CLearner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy

class CriticNet(nn.Module):   #此处的Module不是自写的文件Modules,
    def __init__(self, nr_actions, nr_agents, state_shape, observation_shape, history_length, nr_hidden_layers=128):
        super(CriticNet, self).__init__()
        self.nr_actions = nr_actions
        self.nr_agents = nr_agents
        self.local_observation_shape = int(numpy.prod(observation_shape)*history_length)
        self.global_input_shape = numpy.prod(state_shape)
        self.input_shape = \
            int(self.nr_actions*self.nr_agents)+\
            self.global_input_shape+\
            self.local_observation_shape+\
            self.nr_agents

        # Set up network layers 设置网络层
        self.fc1 = nn.Linear(self.input_shape, nr_hidden_layers)
        self.fc2 = nn.Linear(nr_hidden_layers, nr_hidden_layers)
        self.fc3 = nn.Linear(nr_hidden_layers, self.nr_actions)

    def forward(self, states, histories, actions, device):
        inputs = self.build_inputs(states, histories, actions, device)
        x = F.elu(self.fc1(inputs))
        x = F.elu(self.fc2(x))
        q = self.fc3(x)
        return q

    def build_inputs(self, states, histories, actions, device):
        batch_size = states.size(0)
        new_batch_size = batch_size*self.nr_agents
        states = states.view(batch_size, -1)
        states_ = []
        for state in states:
            states_ += [state.numpy() for _ in range(self.nr_agents)]
        states = torch.tensor(states_, device=device, dtype=torch.float32)
        histories = histories.view(new_batch_size, -1)
        agent_ids = numpy.concatenate(\
            [numpy.eye(self.nr_agents) for _ in range(batch_size)])
        agent_ids = torch.tensor(agent_ids, device=device, dtype=torch.float32)
        actions = actions.view(-1, self.nr_agents)
        joint_actions = []
        for joint_action in actions:
            for i in range(self.nr_agents):
                masked_joint_action = numpy.zeros(self.nr_agents*self.nr_actions)
                for j in range(self.nr_agents):
                    action = joint_action[j]
                    if i != j:
                        masked_joint_action[j*self.nr_actions + action] = 1
                joint_actions.append(masked_joint_action)
        joint_actions = torch.tensor(joint_actions, device=device, dtype=torch.float32)
        assert states.size(0) == new_batch_size, "Wanted {} but got shape {}".format(new_batch_size, states.size())
        assert states.size(1) == self.global_input_shape, "Wanted {} but got shape {}".format(self.global_input_shape, states.size())
        assert histories.size(0) == new_batch_size, "Wanted {} but got shape {}".format(new_batch_size, histories.size())
        assert histories.size(1) == self.local_observation_shape, "Wanted {} but got shape {}".format(self.local_observation_shape, histories.size())
        assert agent_ids.size(0) == new_batch_size, "Wanted {} but got shape {}".format(new_batch_size, agent_ids.size())
        assert agent_ids.size(1) == self.nr_agents, "Wanted {} but got shape {}".format(self.nr_agents,agent_ids.size())
        assert joint_actions.size(0) == new_batch_size, "Wanted {} but got shape {}".format(joint_actions.size())
        assert joint_actions.size(1) == int(self.nr_actions*self.nr_agents), "Wanted {} but got shape {}".format(int(self.nr_actions*self.nr_agents), joint_actions.size())
        inputs = torch.cat([states, histories, joint_actions, agent_ids], dim=-1)
        assert inputs.size(0) == new_batch_size, "Wanted {} but got shape {}".format(new_batch_size, inputs.size())
        assert inputs.size(1) == self.input_shape, "Wanted {} but got shape {}".format(self.input_shape, inputs.size())
        return inputs

class COMALearner(A2CLearner):

    def __init__(self, params):
        params["adversary_ratio"] = 0
        super(COMALearner, self).__init__(params)
        self.critic_net = CriticNet(self.nr_actions, \
            self.nr_agents, self.global_input_shape, self.input_shape, self.max_history_length)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.alpha)

    def value_update(self, minibatch_data, is_adversary):
        if not is_adversary:
            states = minibatch_data["states"]
            histories = minibatch_data["pro_histories"]
            returns = minibatch_data["pro_returns"]
            actions = minibatch_data["pro_actions"]
            Q_values = self.critic_net(states, histories, actions, self.device)\
                .gather(1, actions.unsqueeze(1)).squeeze()
            self.critic_optimizer.zero_grad()
            loss = F.mse_loss(Q_values, returns)
            loss.backward()
            self.critic_optimizer.step()

    def joint_action_probs_(self, histories):
        action_probs = []
        for i, agent_id in enumerate(range(histories.shape[1])):
            history = [[joint_obs[i]] for joint_obs in histories]
            history = torch.tensor(history, device=self.device, dtype=torch.float32)
            is_adversary = agent_id in self.adversary_ids
            probs, value = self.policy_net(history, is_adversary)
            assert len(probs) == 1, "Expected length 1, but got shape {}".format(probs.shape)
            probs = probs.detach().numpy()[0]
            value = value.detach()
            action_probs.append(probs)
        return action_probs

    def policy_update(self, minibatch_data, optimizer, is_adversary, random_agent_index=None):
        policy_loss = 0
        if not is_adversary:
            states = minibatch_data["states"]
            next_states = minibatch_data["next_states"]
            actions = minibatch_data["pro_actions"]
            histories = minibatch_data["pro_histories"]
            next_histories = minibatch_data["next_pro_histories"]
            rewards = minibatch_data["pro_rewards"]
            next_probs = self.joint_action_probs_(next_histories.numpy())
            next_actions = torch.tensor([numpy.random.choice(self.actions, p=p) for p in next_probs], device=self.device, dtype=torch.long)
            action_probs, _ = self.policy_net(histories, is_adversary)
            expected_Q_values = self.critic_net(states, histories, actions, self.device).detach()
            next_expected_Q_values = self.critic_net(next_states, next_histories, next_actions, self.device).detach()
            policy_losses = []
            for probs, next_probs, action, Q_values, next_Q, R in\
                zip(action_probs, next_probs, actions, expected_Q_values, next_expected_Q_values, rewards):
                baseline = sum(probs*Q_values)
                baseline = baseline.detach()
                next_Q = sum(next_probs*next_Q.numpy())
                advantage = R + self.gamma*next_Q - baseline
                m = Categorical(probs)
                policy_losses.append(-m.log_prob(action)*advantage)
            policy_loss = torch.stack(policy_losses).mean()
            loss = policy_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return True
