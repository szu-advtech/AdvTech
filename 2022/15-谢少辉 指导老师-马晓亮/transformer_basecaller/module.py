from torch import nn
from torch.nn import GLU


class FeedForwardModule(nn.Module):
    def __init__(self, d_model, d_ff, dropout_p):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.layers(x)


class DepthwiseConv1d(nn.Module):
    """
    When groups == in_channels and out_channels == K * in_channels, where K is a positive integer,
    this operation is termed in literature as depthwise convolution.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by depthwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = False,
    ) -> None:
        super().__init__()
        assert out_channels % in_channels == 0, "out_channels should be constant multiple of in_channels"
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            groups=in_channels,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class PointwiseConv1d(nn.Module):
    """
    When kernel size == 1 conv_1d, this operation is termed in literature as pointwise convolution.
    This operation often used to match dimensions.

    Args:
        in_channels (int): Number of channels in the input
        out_channels (int): Number of channels produced by the convolution
        stride (int, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        bias (bool, optional): If True, adds a learnable bias to the output. Default: True

    Inputs: inputs
        - **inputs** (batch, in_channels, time): Tensor containing input vector

    Returns: outputs
        - **outputs** (batch, out_channels, time): Tensor produces by pointwise 1-D convolution.
    """
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            padding: int = 0,
            bias: bool = True,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=padding,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class Transpose(nn.Module):
    """ Wrapper class of torch.permute() for Sequential module. """
    def __init__(self, shape: tuple):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x.permute(*self.shape)


class ConvolutionModule(nn.Module):
    """
    Conformer convolution module starts with a pointwise convolution and a gated linear unit (GLU).
    This is followed by a single 1-D depthwise convolution layer. Batchnorm is  deployed just after the convolution
    to aid training deep models.

    Args:
        in_channels (int): Number of channels in the input
        kernel_size (int or tuple, optional): Size of the convolution kernel Default: 31

    Inputs: inputs
        inputs  (T, B, C)

    Outputs: outputs
        outputs (T, B, C)
    """
    def __init__(
            self,
            in_channels: int,
            kernel_size: int = 31,
            expansion_factor: int = 2,
    ) -> None:
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size should be a odd number for 'SAME' padding"
        assert expansion_factor == 2, "Currently, Only Supports expansion_factor 2"

        self.sequential = nn.Sequential(
            Transpose(shape=(1, 2, 0)), # T x B x C => B x C x T
            PointwiseConv1d(in_channels, in_channels * expansion_factor, stride=1, padding=0, bias=True),
            GLU(dim=1),
            DepthwiseConv1d(in_channels, in_channels, kernel_size, stride=1, padding=(kernel_size - 1) // 2),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(), # Swish(),
            PointwiseConv1d(in_channels, in_channels, stride=1, padding=0, bias=True),
            Transpose(shape=(2, 0, 1))  # B x C x T => T x B x C
        )

    def forward(self, x):
        return self.sequential(x)
