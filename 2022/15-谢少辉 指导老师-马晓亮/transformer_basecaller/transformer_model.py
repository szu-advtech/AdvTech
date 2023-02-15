import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import MultiheadAttention
from .module import FeedForwardModule, ConvolutionModule


class TransformerEncoderLayer(nn.Module):
    """
    参考实现:
        [1] https://nlp.seas.harvard.edu/2018/04/01/attention.html
        [2] https://github.com/openspeech-team/openspeech/tree/main/openspeech/encoders/transformer_encoder.py
    """
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


class ConvolutionTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        d_ff,
        num_heads,
        dropout
    ):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.self_attn = MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout,)
        self.dropout1 = nn.Dropout(dropout)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.conv = ConvolutionModule(in_channels=d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.layer_norm3 = nn.LayerNorm(d_model)
        self.feed_forward = FeedForwardModule(d_model, d_ff, dropout)
        self.dropout3 = nn.Dropout(dropout)

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
        x = self.conv(x)
        x = self.dropout2(x)
        x += residual

        residual = x
        x = self.layer_norm3(x)
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x += residual
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for Transformer models

    Non-trainable pre-defined position encoding based on sinus and cosinus waves.

    Copied from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
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


class SACallModel(nn.Module):
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
        use_conv_transformer_encoder=False,
    ):
        super(SACallModel, self).__init__()
        """
        Args:
            convolution (nn.Module): module with: in [batch, channel, len]; out [batch, channel, len]
            pe (nn.Module): positional encoding module with: in [len, batch, channel]; out [len, batch, channel]
            encoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
            decoder (nn.Module): module with: in [len, batch, channel]; out [len, batch, channel]
        """
        def build_cnn():
            cnn = nn.Sequential(
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

            return cnn

        def build_encoder():
            pe = PositionalEncoding(d_model, dropout, max_len=4000)
            if not use_conv_transformer_encoder:
                transformer_layers = nn.ModuleList([
                    # TransformerEncoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout)
                    torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads, dim_feedforward=d_ff, dropout=dropout)
                    for _ in range(num_layers)
                ])
            else:
                transformer_layers = nn.ModuleList([
                    ConvolutionTransformerEncoderLayer(d_model=d_model, d_ff=d_ff, num_heads=num_heads, dropout=dropout)
                    for _ in range(num_layers)
                ])
            transformer_layers = nn.Sequential(*transformer_layers)
            encoder = nn.Sequential(pe, transformer_layers)
            return encoder

        self.convolution = build_cnn()
        self.encoder = build_encoder()
        self.decoder = nn.Linear(d_model, alphabet_size)

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
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    @staticmethod
    def get_logits(net_output, blank_weight: float = 0, blank_mode: str = 'add'):
        logits = net_output
        if blank_weight != 0:
            if blank_mode == "add":
                logits[..., 0] += blank_weight
            elif blank_mode == "set":
                logits[..., 0] = blank_weight
            else:
                raise Exception(f"invalid blank mode {blank_mode}")
        return logits

    @staticmethod
    def get_normalized_probs(
            net_output,
            log_probs: bool = True,
            blank_weight: float = 0,
            blank_mode: str = 'add'
    ):
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = SACallModel.get_logits(net_output, blank_weight=blank_weight, blank_mode=blank_mode)

        if log_probs:
            return F.log_softmax(logits.float(), dim=-1, dtype=torch.float32)
        else:
            return F.softmax(logits.float(), dim=-1, dtype=torch.float32)
