import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention
from .module import FeedForwardModule


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_heads,
        dropout
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardModule(d_model, d_ff, dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: T x B x C
        """
        residual = x
        x = self.layer_norm1(x)
        x, attn_weights = self.self_attn(query=x, key=x, value=x)
        x = self.dropout1(x)
        x += residual

        residual = x
        x = self.layer_norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x += residual
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.0, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: T x B x C
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class InterCTCModel(nn.Module):
    def __init__(
        self,
        d_model=256,
        kernel=3,
        stride=2,
        padding=1,
        dilation=1,
        num_layers=6,
        num_heads=8,
        d_ff=1024,
        dropout=0.1,
        alphabet_size=5,
    ):
        super().__init__()

        self.convolution = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=d_model // 2,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False),
            nn.BatchNorm1d(num_features=d_model // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                in_channels=d_model // 2,
                out_channels=d_model,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False),
            nn.BatchNorm1d(num_features=d_model),
            nn.ReLU(inplace=True)
        )
        self.pe = PositionalEncoding(d_model, dropout, max_len=4000)
        self.transformer_layers = nn.ModuleList([
            # TransformerEncoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout)
            torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc1 = nn.Linear(d_model, alphabet_size, bias=False)
        self.fc2 = nn.Linear(alphabet_size, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x (tensor) : [batch, len]
        Return:
            T x B x C
        """
        x = x.unsqueeze(1)      # BxT => BxCxT
        x = self.convolution(x)

        x = x.permute(2, 0, 1)  # BxCxT => TxBxC
        output_lengths = x.new_full((x.size(1),), x.size(0), dtype=torch.long)

        ret = []
        x = self.pe(x)
        for layer in self.transformer_layers:
            x = layer(x)
            x = self.layer_norm(x)
            log_prob = F.log_softmax(self.fc1(x), dim=-1)
            ret.append(log_prob)
            x = x + self.fc2(torch.exp(log_prob))
        return ret, output_lengths


def intermediate_ctc_loss(criterion, logp_array, target, output_len, target_len, intermediate_loss_weight=0.5):
    intermediate_loss_factor = 1.0 / (len(logp_array) - 1)
    intermediate_loss = criterion(logp_array[0], target, output_len, target_len)
    for i in range(1, len(logp_array) - 1):
        intermediate_loss = intermediate_loss + criterion(logp_array[i], target, output_len, target_len)
    final_loss = criterion(logp_array[len(logp_array) - 1], target, output_len, target_len)
    return (1.0 - intermediate_loss_weight) * final_loss + intermediate_loss_weight * intermediate_loss_factor * intermediate_loss

