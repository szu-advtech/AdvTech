import numpy as np
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F


class ActionModel(nn.Module):
    def __init__(self, action_size, feature_size, hidden_size, layers, dist='tanh_normal',
                 activation=nn.ELU, min_std=1e-4, init_std=5, mean_scale=5, device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()
        self.action_size = action_size
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.layers = layers
        self.dist = dist
        self.activation = activation
        self.min_std = min_std
        self.init_std = init_std
        self.mean_scale = mean_scale
        self.feedforward_model = self.build_model()
        self.raw_init_std = np.log(np.exp(self.init_std) - 1) #噪声方差？
        self.device = device
        self.to(device)

    def build_model(self):
        model = [nn.Linear(self.feature_size, self.hidden_size)]
        model += [self.activation()]
        for i in range(1, self.layers):
            model += [nn.Linear(self.hidden_size, self.hidden_size)]
            model += [self.activation()]
        if self.dist == 'tanh_normal':
            model += [nn.Linear(self.hidden_size, self.action_size * 2)]
        else:
            raise NotImplementedError(f'{self.dist} not implemented')
        return nn.Sequential(*model)

    def forward(self, state_features):
        x = self.feedforward_model(state_features) #输出action size*2
        dist = None
        if self.dist == 'tanh_normal':
            mean, std = torch.chunk(x, 2, -1) #拆分输出
            mean = self.mean_scale * torch.tanh(mean / self.mean_scale) #标准化动作mean
            std = F.softplus(std + self.raw_init_std) + self.min_std #计算方差，保证方差在一个阈值
            dist = D.Normal(mean, std)
            dist = D.TransformedDistribution(dist, TanhBijector())
            dist = D.Independent(dist, 1)
            dist = SampleDist(dist) #返回抽样的动作样本
        return dist


class TanhBijector(D.Transform):
    def __init__(self):
        super().__init__()
        self.bijective = True

    @property
    def sign(self):
        return 1.

    def _call(self, x):
        return torch.tanh(x)

    def _inverse(self, y: torch.Tensor):
        y = torch.where(
            (torch.abs(y) <= 1.),
            torch.clamp(y, -0.99999997, 0.99999997),
            y
        )

        y = atanh(y)
        return y

    def log_abs_det_jacobian(self, x, y):
        return 2. * (np.log(2) - x - F.softplus(-2. * x))


class SampleDist:
    def __init__(self, dist: D.Distribution, samples=100):
        self._dist = dist
        self._samples = samples

    @property
    def name(self):
        return 'SampleDist'

    def __getattr__(self, name):
        return getattr(self._dist, name)

    def mean(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        return torch.mean(sample, 0)

    def mode(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        batch_size = sample.size(1)
        feature_size = sample.size(2)
        indices = torch.argmax(logprob, dim=0).reshape(1, batch_size, 1).expand(1, batch_size, feature_size)
        return torch.gather(sample, 0, indices).squeeze(0)

    def entropy(self):
        dist = self._dist.expand((self._samples, *self._dist.batch_shape))
        sample = dist.rsample()
        logprob = dist.log_prob(sample)
        return -torch.mean(logprob, 0)

    def sample(self):
        return self._dist.sample()


def atanh(x):
    return 0.5 * torch.log((1 + x) / (1 - x))


class DenseModel(nn.Module):
    def __init__(self, feature_size: int, output_shape: tuple, layers: int, hidden_size: int, dist='normal',
                 activation=nn.ELU):
        super().__init__()
        self._output_shape = output_shape
        self._layers = layers
        self._hidden_size = hidden_size
        self._dist = dist
        self.activation = activation
        self._feature_size = feature_size
        self.model = self.build_model()

    def build_model(self):
        model = [nn.Linear(self._feature_size, self._hidden_size)]
        model += [self.activation()]
        for i in range(self._layers - 1):
            model += [nn.Linear(self._hidden_size, self._hidden_size)]
            model += [self.activation()]
        model += [nn.Linear(self._hidden_size, int(np.prod(self._output_shape)))]
        return nn.Sequential(*model)

    def forward(self, features):
        dist_inputs = self.model(features)
        reshaped_inputs = torch.reshape(dist_inputs, features.shape[:-1] + self._output_shape)
        if self._dist == 'normal':
            return D.independent.Independent(D.Normal(reshaped_inputs, 1), len(self._output_shape))
        if self._dist == 'binary':
            return D.independent.Independent(D.Bernoulli(logits=reshaped_inputs), len(self._output_shape))
        raise NotImplementedError(self._dist)