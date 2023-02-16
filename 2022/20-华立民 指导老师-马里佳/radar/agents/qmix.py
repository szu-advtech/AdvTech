from radar.agents.vdn import VDNLearner
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from radar.utils import get_param_or_default

class QMIXNetwork(nn.Module):
    def __init__(self, input_shape, nr_agents, mixing_hidden_size=128):
        super(QMIXNetwork, self).__init__()
        self.nr_agents = nr_agents
        self.mixing_hidden_size = mixing_hidden_size
        self.state_shape = numpy.prod(input_shape)
        self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_shape, mixing_hidden_size),
                                           nn.ELU(),
                                           nn.Linear(mixing_hidden_size, mixing_hidden_size * self.nr_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(self.state_shape, mixing_hidden_size),
                                           nn.ELU(),
                                           nn.Linear(mixing_hidden_size, mixing_hidden_size))
        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_shape, mixing_hidden_size)
        # V(s) instead of a bias for the last layers
        self.V = nn.Sequential(nn.Linear(self.state_shape, mixing_hidden_size),
                               nn.ELU(),
                               nn.Linear(mixing_hidden_size, 1))

    def forward(self, global_state, Q_values):
        global_state = global_state.view(global_state.size(0), -1)
        w1 = torch.abs(self.hyper_w_1(global_state))
        b1 = self.hyper_b_1(global_state)
        w1 = w1.view(-1, self.nr_agents, self.mixing_hidden_size)
        b1 = b1.view(-1, 1, self.mixing_hidden_size)
        hidden = F.elu(torch.bmm(Q_values, w1) + b1)
        # Second layer
        w_final = torch.abs(self.hyper_w_final(global_state))
        w_final = w_final.view(-1, self.mixing_hidden_size, 1)
        # State-dependent bias
        v = self.V(global_state).view(-1, 1, 1)
        # Compute final output
        y = torch.bmm(hidden, w_final) + v
        return y.view(Q_values.size(0), -1, 1)

class QMIXLearner(VDNLearner):

    def __init__(self, params):
        self.global_input_shape = params["global_observation_shape"]
        self.nr_agents = params["nr_agents"]
        self.device = torch.device("cpu")
        super(QMIXLearner, self).__init__(params)
        self.global_value_network = self.make_mixer_neural_network()
        self.global_target_network = self.make_mixer_neural_network()
        parameters = list(self.policy_net.parameters()) + list(self.global_value_network.parameters())
        self.protagonist_optimizer = torch.optim.Adam(parameters, lr=self.alpha)

    def make_mixer_neural_network(self):
        return QMIXNetwork(self.global_input_shape, self.nr_agents).to(self.device)

    def update_target_network(self, loss=0):
        if self.training_count % self.target_update_period is 0 and self.global_target_network is not None:
            super(QMIXLearner, self).update_target_network()
            self.global_target_network.load_state_dict(self.global_value_network.state_dict())
            self.global_target_network.eval()

    def global_value(self, network, Q_values, states):
        Q_values = Q_values.view(-1, 1, self.nr_agents)
        return network(states, Q_values).squeeze()