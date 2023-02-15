import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.utils.data as Data


class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM, self).__init__()
        in_dim = 9
        hidden_dim = 16
        self.lstm1 = nn.LSTM(in_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.activate1 = nn.Tanh()
        self.lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim * 2, bidirectional=True, batch_first=True)
        self.activate2 = nn.Tanh()
        self.linear = nn.Linear(hidden_dim * 4, 64)
        self.activate3 = nn.Tanh()
        self.head = nn.Linear(64, 27)

    def forward(self, x):
        """
        x: [batch_size, seq_len, input_dim]
        """
        bs, _, _ = x.shape
        # final_hidden_state, final_cell_state : [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        output, (_, _) = self.lstm1(x)
        output = self.activate1(output)
        output, (final_hidden_state, final_cell_state) = self.lstm2(output)
        final_hidden_state = self.activate2(final_hidden_state)
        output = final_hidden_state.transpose(0, 1).reshape(bs, -1)
        output = self.linear(output)
        output = self.activate3(output)
        output = self.head(output)
        return output


if __name__ == '__main__':
    model = BiLSTM()
    x = torch.randn([64, 10, 9])
    model(x)