import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple
from copy import deepcopy
import math
import numpy as np

from nets.encoders import video_encoder
from nets.EFNet import MLP
from nets.EFKptsSDNet import SDNet
from nets.EFGCNTmp import kpts_decoder_temporal


class NRIMLP(nn.Module):
    """
    MLP uses by the NRI network
    Copied from https://github.com/ethanfetaya/NRI

    Attributes
    ----------
    n_in: int, dimension of input embeddings
    n_hid: int, dimension of hidden embeddings
    n_out: int, dimension of output embeddings
    do_prob: float, dropout used in between layers

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 n_in: int,
                 n_hid: int,
                 n_out: int,
                 do_prob: float = 0.):
        """
        :param n_in: int, dimension of input embeddings
        :param n_hid: int, dimension of hidden embeddings
        :param n_out: int, dimension of output embeddings
        :param do_prob: float, dropout used in between layers
        """

        super().__init__()

        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)
        self.bn = nn.BatchNorm1d(n_out)
        self.dropout_prob = do_prob

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights for linear and batch norm layers
        """

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self,
                   inputs: torch.tensor) -> torch.tensor:
        """
        Batch normalization after reshaping input to correct shape
        :param inputs: torch.tensor, input tensor of shape N*num_frames*1*d
        :return: Tensor of shape N*num_frames*1*d
        """

        x = inputs.view(inputs.size(0) * inputs.size(1), -1)
        x = self.bn(x)
        return x.view(inputs.size(0), inputs.size(1), -1)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor of shape N*whatever*n_in
        :return: Tensor of shape N*whatever*n_out
        """

        x = F.elu(self.fc1(x))
        x = F.dropout(x, self.dropout_prob, training=self.training)
        x = F.elu(self.fc2(x))
        return self.batch_norm(x)


class Conv3DResBlock(nn.Module):
    """
    3D convolution block with residual connections

    Attributes
    ----------
    conv: torch.nn.Conv3d, PyTorch Conv3D model
    bn: torch.nn.BatchNorm3d, PyTorch 3D batch normalization layer
    pool: torch.nn.AvgPool3d, PyTorch average 3D pooling layer
    dropout: torch.nn.Dropout3D, PyTorch 3D dropout layer
    shortcut: torch.nn.Conv3d, pyTorch 1*1 conv model to equalize the number of channels for residual addition

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 in_channels: int,
                 padding: int,
                 out_channels: int,
                 kernel_size: int,
                 pool_size: int,
                 cnn_dropout_p: float = 0.0):
        """
        :param in_channels: int, number of input channels
        :param padding: int, 0 padding dims
        :param out_channels: int, number of filters to use
        :param kernel_size: int, filter size
        :param pool_size: int, pooling kernel size for the spatial dims
        :param cnn_dropout_p: float, cnn dropout rate
        """

        super().__init__()

        # 1x1 convolution to make the channels equal for the residual
        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv3d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=1)

        self.conv = nn.Conv3d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              padding=(padding, padding, padding))
        self.bn = nn.BatchNorm3d(out_channels)
        self.pool = nn.AvgPool3d(kernel_size=(1, pool_size, pool_size))
        self.dropout = nn.Dropout3d(p=cnn_dropout_p)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor of shape N*1*64*T*H*W
        :return: Tensor of shape N*out_channels*T*H'*W'
        """

        if self.shortcut is not None:
            residual = self.shortcut(x)
        else:
            residual = x

        x = self.conv(x)
        x = self.bn(x)
        x = x + residual
        x = self.pool(x)
        x = F.elu(x)

        return self.dropout(x)


class Custom3DConv(nn.Module):
    """
    3D convolution network

    Attributes
    ----------
    conv: torch.nn.Sequential, the convolutional network containing residual blocks
    output_fc: torch.nn.Sequential, the FC layer applied to the output of convolutional network

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 out_channels: list = None,
                 kernel_sizes: list = None,
                 pool_sizes: list = None,
                 output_dim: int = 128,
                 cnn_dropout_p: float = 0):
        """
        :param out_channels: list, output channels for each layer
        :param kernel_sizes: list, kernel sizes for each layer
        :param pool_sizes: list, pooling kernel sizes for each layer
        :param output_dim: int, dimension of output embeddings
        :param cnn_dropout_p: float, dropout ratio of the CNN
        """

        super().__init__()

        n_conv_layers = len(out_channels)

        # Default list arguments
        if kernel_sizes is None:
            kernel_sizes = [3]*n_conv_layers
        if pool_sizes is None:
            pool_sizes = [2]*n_conv_layers

        # Ensure input params are list
        if type(out_channels) is not list:
            out_channels = [out_channels]*n_conv_layers
        else:
            assert len(
                out_channels) == n_conv_layers, 'Provide channel parameter for all layers.'
        if type(kernel_sizes) is not list:
            kernel_sizes = [kernel_sizes]*n_conv_layers
        else:
            assert len(
                kernel_sizes) == n_conv_layers, 'Provide kernel size parameter for all layers.'
        if type(pool_sizes) is not list:
            pool_sizes = [pool_sizes]*n_conv_layers
        else:
            assert len(
                pool_sizes) == n_conv_layers, 'Provide pool size parameter for all layers.'

        # Compute paddings to preserve temporal dim
        paddings = list()
        for kernel_size in kernel_sizes:
            paddings.append(math.floor((kernel_size - 1) / 2))

        # Conv layers
        convs = list()

        # Add first layer
        convs.append(nn.Sequential(Conv3DResBlock(in_channels=1,
                                                  padding=paddings[0],
                                                  out_channels=out_channels[0],
                                                  kernel_size=kernel_sizes[0],
                                                  pool_size=pool_sizes[0],
                                                  cnn_dropout_p=cnn_dropout_p)))

        # Add subsequent layers
        for layer_num in range(1, n_conv_layers):
            convs.append(nn.Sequential(Conv3DResBlock(in_channels=out_channels[layer_num-1],
                                                      padding=paddings[layer_num],
                                                      out_channels=out_channels[layer_num],
                                                      kernel_size=kernel_sizes[layer_num],
                                                      pool_size=pool_sizes[layer_num],
                                                      cnn_dropout_p=cnn_dropout_p)))
        # Change to sequential
        self.conv = nn.Sequential(*convs)

        # Output linear layer
        self.output_fc = nn.Sequential(nn.AdaptiveAvgPool3d((None, 1, 1)),
                                       nn.Flatten(start_dim=2),
                                       nn.Linear(out_channels[-1], output_dim),
                                       nn.ReLU(inplace=True))

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor of shape N*1*T*H*W
        :return: Tensor of shape N*T*output_dim
        """

        # CNN layers
        x = self.conv(x).permute(0, 2, 1, 3, 4)

        # FC layer
        x = self.output_fc(x)

        return x


EMBEDDERS = {'3dconv': Custom3DConv}


class PositionalEncoding(nn.Module):
    """
    Positional encoding

    Copied from https://pytorch.org/tutorials/beginner/transformer_tutorial.html and modified.

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 d_model,
                 max_len=250):
        """
        :param d_model: int, embeddings' dimension
        :param max_len: int, maximum sequence length to support
        """

        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor of shape N*T*d_model
        :return: Tensor of shape N*T*d_model
        """

        x = x + self.pe[:x.size(1)].permute(1, 0, 2)

        return x


class VideoEncoder(nn.Module):
    """
    Video Encoder model

    Attributes
    ----------
    config: dict, Video Encoder config dictionary
    add_positional_embeddings: bool, indicates whether positional embeddings are added to frame embeddings
    embedder_type: str, indicates the network used for the Video Encoder
    embedder: torch.nn.Module, PyTorch model for the Video Encoder
    positional_encoder: torch.nn.Module, PyTorch module to produce positional embeddings
    dropout: float, dropout used after adding positional embeddings

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 config: dict = None):
        """
        :param config: dict, Video Encoder config dictionary
        """

        super().__init__()

        embedder_config = deepcopy(config)
        name = embedder_config.pop('name')
        embedding_dim = embedder_config['output_dim']
        embedding_dropout = embedder_config.pop('fc_dropout_p')
        self.add_positional_embeddings = embedder_config.pop(
            'add_positional_embeddings')

        assert name in ['3dconv'], \
            'Embedder type must be one of [3dconv]'

        self.embedder_type = name

        self.embedder = EMBEDDERS[name](**embedder_config)
        self.positional_encoder = PositionalEncoding(d_model=embedding_dim)
        self.dropout = nn.Dropout(p=embedding_dropout)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor of shape N*1*T*H*W
        :return: Tensor of shape N*T*output_dim
        """

        x = self.embedder(x=x)
        print(self.embedder)

        # Add positional embeddings
        if self.add_positional_embeddings:
            x = self.positional_encoder(x)

        return self.dropout(x)


class MLPEdgeEncoder(nn.Module):
    """
    NRI network to find weights over edges
    Copied from and modified: https://github.com/ethanfetaya/NRI

    Attributes
    ----------
    mlp1: torch.nn.Module, initial MLP transforming node embeddings
    mlp2: torch.nn.Module, first edge->node MLP
    mlp3: torch.nn.Module, first node->edge MLP
    mlp4: torch.nn.Module, transform edge embeddings with skip connection
    fc_out: torch.nn.Module, output linear layer to map embeddings to weights (would be weights after Sigmoid)
    activation_func: torch.nn.Sigmoid, output Sigmoid activation function
    num_vids_per_sample: int, number of videos per sample
    num_frames: int, number of frames per clip
    rel_rec: torch.tensor, the matrix aggregating receiving embeddings
    rel_send: torch.tensor, the matrix aggregating outgoing embeddings

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 input_dim: int = 128,
                 fc_dropout_p: float = 0.,
                 hidden_dim: int = 100,
                 num_frames: int = 32,
                 num_vids_per_sample: int = 1,
                 device: torch.device = torch.device('cpu')):
        """
        :param input_dim: int, input node embeddings dim
        :param fc_dropout_p: float, dropout rate
        :param hidden_dim: int, node hidden embedding dim
        :param num_frames: int, number of frames per clip
        :param num_vids_per_sample: int, number of videos per patient
        :param device: torch.device, device for the model
        """

        super().__init__()

        self.mlp1 = NRIMLP(input_dim, hidden_dim, hidden_dim, fc_dropout_p)
        self.mlp2 = NRIMLP(hidden_dim * 2, hidden_dim,
                           hidden_dim, fc_dropout_p)
        self.mlp3 = NRIMLP(hidden_dim, hidden_dim, hidden_dim, fc_dropout_p)
        self.mlp4 = NRIMLP(hidden_dim * 3, hidden_dim,
                           hidden_dim, fc_dropout_p)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.activation_func = nn.Sigmoid()
        self.num_vids_per_sample = num_vids_per_sample
        self.num_frames = num_frames

        # Create matrices fetching outgoing and incoming node messages
        off_diag = np.ones([num_frames * num_vids_per_sample,
                            num_frames * num_vids_per_sample]) - np.eye(num_frames * num_vids_per_sample)
        rel_rec = np.array(self._encode_onehot(
            np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(self._encode_onehot(
            np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)

        self._init_weights()

    @staticmethod
    def _encode_onehot(labels: np.ndarray) -> np.ndarray:
        """
        Changes labels to one_hot format
        Code is copied from https://github.com/jwzhanggy/GResNet

        :param labels: numpy.ndarray, labels to change to one_hot format
        :return: one_hot encoded labels
        """

        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def _init_weights(self):
        """
        Initialize weights for linear layers
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _edge2node(self, x: torch.tensor, rel_rec: torch.tensor) -> torch.tensor:
        """
        Performs edge to node feature transformation

        :param x: torch.tensor, input tensor of shape N*(num_frames*num_vids_per_samples*(num_frames*num_vids_per_samples-1))*d
        :param rel_rec: torch.tensor, matrix used to gather incoming node features
        :return: Torch tensor of shape N*(num_frames*num_vids_per_samples)*d
        """

        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)

        return incoming / incoming.size(1)

    def _node2edge(self, x: torch.tensor, rel_rec: torch.tensor, rel_send: torch.tensor) -> torch.tensor:
        """
        Performs node to edge feature transformation

        :param x: torch.tensor, input tensor of shape N*(num_frames*num_vids_per_samples)*d
        :param rel_rec: torch.tensor, matrix used to gather incoming node features
        :param rel_send: torch.tensor, matrix used to gather outgoing node features
        :return: Torch tensor of shape N*(num_frames*num_vids_per_samples*(num_frames*num_vids_per_samples-1))*d
        """

        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)

        return edges

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Model's forward propagation

        :param x: torch.tensor, input tensor of shape [N*num_frames*num_vids_per_sample*num_clips_per_vid, input_dim]
        :return: Torch tensor of shape N*(num_frames*num_vids_per_samples*(num_frames*num_vids_per_samples-1))*1
        """

        x = x.view(x.size(0) // (self.num_frames * self.num_vids_per_sample),
                   self.num_frames * self.num_vids_per_sample, 1, -1)

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self._node2edge(x, self.rel_rec, self.rel_send)
        x = self.mlp2(x)
        x_skip = x

        x = self._edge2node(x, self.rel_rec)
        x = self.mlp3(x)
        x = self._node2edge(x, self.rel_rec, self.rel_send)
        x = torch.cat((x, x_skip), dim=2)  # Skip connection
        x = self.mlp4(x)

        return self.activation_func(self.fc_out(x))


class MLPNodeEncoder(nn.Module):
    """
    NRI network to find weights over nodes
    Copied from and modified: https://github.com/ethanfetaya/NRI

    Attributes
    ----------
    mlp1: torch.nn.Module, MLP transforming initial node embeddings
    mlp2: torch.nn.Module, first  ->  MLP
    mlp3: torch.nn.Module, second node -> edge MLP
    mlp4: torch.nn.Module, second edge -> node MLP with skip connection
    mlp5: torch.nn.Module, final MLP imposed on node embeddings
    fc_out: torch.nn.Module, output linear layer to map embeddings to weights (would be weights after Sigmoid)
    activation_func: torch.nn.Sigmoid, output Sigmoid activation function
    num_vids_per_sample: int, number of videos per sample
    num_frames: int, number of frames per clip
    rel_rec: torch.tensor, the matrix aggregating receiving embeddings
    rel_send: torch.tensor, the matrix aggregating outgoing embeddings

    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 input_dim: int = 128,
                 fc_dropout_p: float = 0.,
                 hidden_dim: int = 100,
                 num_frames: int = 32,
                 num_vids_per_sample: int = 1,
                 device: torch.device = torch.device('cpu')):
        """
        :param input_dim: int, input node embeddings dim
        :param fc_dropout_p: float, dropout rate
        :param hidden_dim: int, node hidden embedding dim
        :param num_frames: int, number of frames per clip
        :param num_vids_per_sample: int, number of videos per patient
        :param device: torch.device, device for the model
        """

        super().__init__()

        self.mlp1 = NRIMLP(input_dim, hidden_dim, hidden_dim, fc_dropout_p)
        self.mlp2 = NRIMLP(hidden_dim * 2, hidden_dim,
                           hidden_dim, fc_dropout_p)
        self.mlp3 = NRIMLP(hidden_dim, hidden_dim, hidden_dim, fc_dropout_p)
        self.mlp4 = NRIMLP(hidden_dim * 3, hidden_dim,
                           hidden_dim, fc_dropout_p)
        self.mlp5 = NRIMLP(hidden_dim, hidden_dim, hidden_dim, fc_dropout_p)
        self.fc_out = nn.Linear(hidden_dim, 1)
        self.activation_func = nn.Sigmoid()
        self.num_vids_per_sample = num_vids_per_sample
        self.num_frames = num_frames

        # Create matrices fetching outgoing and incoming node messages
        off_diag = np.ones([num_frames * num_vids_per_sample,
                            num_frames * num_vids_per_sample]) - np.eye(num_frames * num_vids_per_sample)
        rel_rec = np.array(self._encode_onehot(
            np.where(off_diag)[0]), dtype=np.float32)
        rel_send = np.array(self._encode_onehot(
            np.where(off_diag)[1]), dtype=np.float32)
        self.rel_rec = torch.FloatTensor(rel_rec).to(device)
        self.rel_send = torch.FloatTensor(rel_send).to(device)

        self._init_weights()

    @staticmethod
    def _encode_onehot(labels: np.ndarray) -> np.ndarray:
        """
        Changes labels to one_hot format
        Code is copied from https://github.com/jwzhanggy/GResNet

        :param labels: numpy.ndarray, labels to change to one_hot format
        :return: one_hot encoded labels
        """

        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                 dtype=np.int32)
        return labels_onehot

    def _init_weights(self):
        """
        Initialize weights for linear layers
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _edge2node(self, x: torch.tensor, rel_rec: torch.tensor) -> torch.tensor:
        """
        Performs edge to node feature transformation

        :param x: torch.tensor, input tensor of shape N*(num_frames*num_vids_per_samples*(num_frames*num_vids_per_samples-1))*d
        :param rel_rec: torch.tensor, matrix used to gather incoming node features
        :return: Torch tensor of shape N*(num_frames*num_vids_per_samples)*d
        """

        # NOTE: Assumes that we have the same graph across all samples.
        incoming = torch.matmul(rel_rec.t(), x)

        return incoming / incoming.size(1)

    def _node2edge(self, x: torch.tensor, rel_rec: torch.tensor, rel_send: torch.tensor) -> torch.tensor:
        """
        Performs node to edge feature transformation

        :param x: torch.tensor, input tensor of shape N*(num_frames*num_vids_per_samples)*d
        :param rel_rec: torch.tensor, matrix used to gather incoming node features
        :param rel_send: torch.tensor, matrix used to gather outgoing node features
        :return: Torch tensor of shape N*(num_frames*num_vids_per_samples*(num_frames*num_vids_per_samples-1))*d
        """

        # NOTE: Assumes that we have the same graph across all samples.
        receivers = torch.matmul(rel_rec, x)
        senders = torch.matmul(rel_send, x)
        edges = torch.cat([senders, receivers], dim=2)

        return edges

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Model's forward propagation

        :param x: torch.tensor, input tensor of shape [N*num_frames*num_vids_per_sample*num_clips_per_vid, input_dim]
        :return: Torch tensor of shape Torch tensor of shape N*num_frames*1
        """

        x = x.view(x.size(0) // (self.num_frames * self.num_vids_per_sample),
                   self.num_frames * self.num_vids_per_sample, 1, -1)

        x = self.mlp1(x)  # 2-layer ELU net per node

        x = self._node2edge(x, self.rel_rec, self.rel_send)
        x = self.mlp2(x)
        x_skip = x

        x = self._edge2node(x, self.rel_rec)
        x = self.mlp3(x)
        x = self._node2edge(x, self.rel_rec, self.rel_send)
        x = torch.cat((x, x_skip), dim=2)  # Skip connection
        x = self.mlp4(x)
        x = self._edge2node(x, self.rel_rec)
        x = self.mlp5(x)

        return self.activation_func(self.fc_out(x))


ENCODERS = {'node': MLPNodeEncoder,
            'edge': MLPEdgeEncoder}


class AttentionEncoder(nn.Module):
    """
    Attention Encoder network to find weights over nodes and edges of echo-graph

    Attributes
    ----------
    device: torch.device, device to use for the model
    num_frames: int, number of frames per clip
    encoder_type: str, indicates whether weights are created over nodes, edges, both or none
    node_encoder: torch.nn.Module, the model for weight distribution over nodes
    edge_encoder: torch.nn.Module, the model for weight distribution over edges


    Methods
    -------
    forward(x): model's forward propagation
    """

    def __init__(self,
                 config: dict):
        """
        :param config: dict, Attention Encoder config dictionary
        """

        super().__init__()

        encoder_config = deepcopy(config)
        name = encoder_config.pop('name')
        self.num_frames = encoder_config['num_frames']
        self.device = encoder_config.pop('device')

        assert name in ['node', 'edge', 'node&edge',
                        'none'], 'encoder type must be one of [node, edge, both, none]'

        self.encoder_type = name

        self.node_encoder = None
        self.edge_encoder = None

        if self.encoder_type == 'node&edge' or self.encoder_type == 'node':
            self.node_encoder = ENCODERS['node'](
                **encoder_config, device=self.device)

        if self.encoder_type == 'node&edge' or self.encoder_type == 'edge':
            self.edge_encoder = ENCODERS['edge'](
                **encoder_config, device=self.device)

    def forward(self, x: torch.tensor) -> torch.tensor:
        """
        Forward propagation

        :param x: torch.tensor, input tensor of shape N*num_frames*d
        :return: tuple of node and edge weights
        """

        x = x.view(x.shape[0]*x.shape[1], -1)

        if self.node_encoder:
            node_z = self.node_encoder(x)
        else:
            node_z = torch.ones(
                (x.shape[0] // self.num_frames, self.num_frames, 1), device=self.device)

        if self.edge_encoder:
            edge_z = self.edge_encoder(x)
        else:
            edge_z = torch.ones((x.shape[0] // self.num_frames, self.num_frames * (self.num_frames - 1), 1),
                                device=self.device)

        return node_z, edge_z


class SDGCN(nn.Module):
    """
    DNN for learning EF. Based on CNN for video encoding into a latent feature vector,
    followed by a sequence of linear layers for regression of EF.
    """

    def __init__(self, config, MLP_channels_ef: List = [16, 32, 32, 48], GCN_channels_kpts: List = [16, 32, 32, 48], is_gpu=False):

        super(SDGCN, self).__init__()

        self.MLP_channels_ef = MLP_channels_ef
        self.GCN_channels_kpts = GCN_channels_kpts
        self.num_frames = 16
        # also include label 0 for 'transition' (or None ES or ED)
        self.num_label_frames = self.num_frames + 1
        self.num_SD_frames = 2  # ES, ED, or 'transition' (None)
        self.img_feature_size = 512
        # self.encoder = video_encoder(backbone=backbone)
        # self.ef_regressor = MLP(features_size=self.encoder.img_feature_size,
        #                         channels=self.MLP_channels_ef, output_channels=1)
        # # self.kpts_regressor = MLP(features_size=self.encoder.img_feature_size + self.num_SD_frames * self.num_label_frames,
        # #                           channels=self.MLP_channels_kpts, output_channels=40*2*2)
        # self.kpts_regressor = kpts_decoder_temporal(features_size=self.encoder.img_feature_size + self.num_SD_frames * self.num_label_frames,
        #                                             kpt_channels=2,
        #                                             gcn_channels=self.GCN_channels_kpts,
        #                                             tsteps=2,
        #                                             num_kpts=40 * 2,
        #                                             is_gpu=is_gpu)

        # self.sd_regressor = SDNet(features_size=self.encoder.img_feature_size,
        #                           num_label_frames=self.num_label_frames, num_sd_frames=self.num_SD_frames)

        self.video_encoder = VideoEncoder(config=config)
        self.attention_encoder = AttentionEncoder(config=config)
        self.kpts_regressor = kpts_decoder_temporal(features_size=self.img_feature_size + self.num_SD_frames * self.num_label_frames,
                                                    kpt_channels=2,
                                                    gcn_channels=self.GCN_channels_kpts,
                                                    tsteps=2,
                                                    num_kpts=40 * 2,
                                                    is_gpu=is_gpu)



    def forward(self, x: torch.Tensor, is_feat: bool = False) -> torch.Tensor:
        # features = self.encoder(x)

        # ef, guiding_layer_ef = self.ef_regressor(features)
        # flatten_sd = self.sd_regressor(features)
        # features_and_sd = torch.cat((features, flatten_sd), dim=1)
        # kpts = self.kpts_regressor(features_and_sd)

        # num_batches = flatten_sd.shape[0]
        # sd = torch.reshape(
        #     flatten_sd, (num_batches, self.num_label_frames, self.num_SD_frames))

        # if is_feat:
        #     return ef, kpts, sd, guiding_layer_ef
        # else:
        #     return ef, kpts, sd

        features = self.video_encoder(x)
        node_weights, edge_weights = self.attention_encoder(features)



if __name__ == '__main__':
    frame_size = 112
    num_frames = 2
    num_kpts = 40
    num_batches = 4
    kpt_channels = 2  # kpts dim
    is_gpu = False
    print('load model')
    img = torch.rand(num_batches, 3, num_frames, frame_size, frame_size)

    config = {
        "video_encoder": {
            "name": "3dconv",
            "out_channels": [16, 32, 64, 128, 256],
            "kernel_sizes": 3,
            "pool_sizes": 2,
            "output_dim": 256,
            "cnn_dropout_p": 0.1,
            "fc_dropout_p": 0.5,
            "add_positional_embeddings": True,

        },
        "attention_encoder": {
            "name": "node&edge",
            "fc_dropout_p": 0.5,
            "hidden_dim": 128,
        }
    }

    if is_gpu:
        img = img.cuda()
    m = SDGCN(config=config, MLP_channels_ef=[
              16, 32, 32, 48], GCN_channels_kpts=[16, 32, 32, 48], is_gpu=is_gpu)
    if is_gpu:
        m = m.cuda()
    print('model loaded')
    ef_pred, kpts_pred, sd_pred = m(img)
    ef, kpts, sd = torch.rand(num_batches, 1), torch.rand(
        num_batches, num_kpts * 2, num_frames), torch.randint(low=0, high=16+1, size=(num_batches, 2))
    if is_gpu:
        ef, kpts, sd = ef.cuda(), kpts.cuda(), sd.cuda()
    loss = [nn.L1Loss(), nn.CrossEntropyLoss()]
    print(loss[0](ef_pred, ef) + loss[0]
          (kpts_pred, kpts) + loss[1](sd_pred, sd))
    optimizer = torch.optim.Adam(m.parameters(), lr=1e-4)
    optimizer.step()
    print("hi")
    pass
