import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from os.path import join

class AdversarialModule(torch.nn.Module):
    def __init__(self, input_shape, outputs, max_history_length, network_constructor):
        super(AdversarialModule, self).__init__()
        self.protagonist_net = network_constructor(input_shape, outputs, max_history_length)
        self.adversary_net = network_constructor(input_shape, outputs, max_history_length)

    def protagonist_parameters(self):
        return self.protagonist_net.parameters()

    def adversary_parameters(self):
        return self.adversary_net.parameters()

    def forward(self, x, is_adversary, use_gumbel_softmax=False):
        if use_gumbel_softmax:
            if is_adversary:
                return self.adversary_net(x, use_gumbel_softmax=True)
            else:
                return self.protagonist_net(x, use_gumbel_softmax=True)
        if is_adversary:
            return self.adversary_net(x)
        else:
            return self.protagonist_net(x)

    def save_weights(self, path):
        protagonist_path = join(path, "protagonist_model.pth")
        torch.save(self.protagonist_net.state_dict(), protagonist_path)
        adversary_path = join(path, "adversary_model.pth")
        torch.save(self.adversary_net.state_dict(), adversary_path)

    def load_weights(self, path):
        protagonist_path = join(path, "protagonist_model.pth")
        self.protagonist_net.load_state_dict(torch.load(protagonist_path, map_location='cpu'))
        self.protagonist_net.eval()
        adversary_path = join(path, "adversary_model.pth")
        self.adversary_net.load_state_dict(torch.load(adversary_path, map_location='cpu'))
        self.adversary_net.eval()

class MLP(nn.Module):

    def __init__(self, input_shape, max_sequence_length, nr_hidden_units=64):
        super(MLP, self).__init__()
        self.input_shape = input_shape
        self.nr_input_features = numpy.prod(self.input_shape)*max_sequence_length
        self.max_sequence_length = max_sequence_length
        self.nr_hidden_units = nr_hidden_units
        if max_sequence_length > 1:
            self.nr_hidden_units = int(2*nr_hidden_units)
        self.fc_net = nn.Sequential(
            nn.Linear(self.nr_input_features, self.nr_hidden_units),
            nn.ELU(),
            nn.Linear(self.nr_hidden_units, self.nr_hidden_units),
            nn.ELU()
        )

    def forward(self, x):
        sequence_length = x.size(0)
        batch_size = x.size(1)
        assert self.max_sequence_length == sequence_length, "Got shape: {}".format(x.size())
        x = x.view(sequence_length, batch_size, -1)
        x = x.permute(1, 0, 2)
        x = torch.reshape(x, (batch_size, -1))
        return self.fc_net(x)