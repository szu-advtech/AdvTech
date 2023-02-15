import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

from dense_models import *
from conv_models import *

class WorldModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        ConvModule = ObservationEncoder

        self.free_nats = 3.
        self.kl_scale = 1.

        self.obs_encoder = ConvModule()

        self.z_encoder = nn.Sequential(
                                nn.Linear(230, 200),
                                nn.ELU(),
                                nn.Linear(200, 200),
                                nn.Tanh(),
                            )
        self.w_contrastive = nn.Sequential(
                                nn.Linear(self.obs_encoder.embed_size, 400),
                                nn.ELU(),
                                nn.Linear(400, 200),
                                nn.Tanh(),
                            )
        obs_modules = [self.z_encoder, self.w_contrastive]

        self._embed_size = self.obs_encoder.embed_size #计算多个特征大小
        
        self.prior = TransitionModel(config['action_size'])
        self.posterior = RepresentationModel(self._embed_size, config['action_size'])
        
        self._hidden_size = 200
        self._deter_size = 200
        self._stoch_size = 30  #隐藏状态维度
        self._feature_size = self._deter_size + self._stoch_size
        self.vae = VAE(self.obs_encoder.embed_size,self._stoch_size)

class TransitionModel(nn.Module):
    def __init__(self, action_size, stochastic_size=30, deterministic_size=200, hidden_size=200, activation=nn.ELU,
                 distribution=D.Normal):
        super().__init__()
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._rnn_input_model = self._build_rnn_input_model()
        self._cell = nn.GRUCell(hidden_size, deterministic_size)
        self._stochastic_prior_model = self._build_stochastic_model()
        self._dist = distribution

    def _build_rnn_input_model(self):
        rnn_input_model = [nn.Linear(self._action_size + self._stoch_size, self._hidden_size)]
        rnn_input_model += [self._activation()]
        return nn.Sequential(*rnn_input_model)

    def _build_stochastic_model(self):  #vae 编码部分
        stochastic_model = [nn.Linear(self._deter_size, self._hidden_size)]
        stochastic_model += [self._activation()]
        stochastic_model += [nn.Linear(self._hidden_size, 2 * self._stoch_size)]
        return nn.Sequential(*stochastic_model)

    def initial_state(self, batch_size, **kwargs):
        return dict(
            mean=torch.zeros(batch_size, self._stoch_size, **kwargs),
            std=torch.zeros(batch_size, self._stoch_size, **kwargs),
            stoch=torch.zeros(batch_size, self._stoch_size, **kwargs),
            deter=torch.zeros(batch_size, self._deter_size, **kwargs),
        )

    def forward(self, prev_action: torch.Tensor, prev_state: dict):
        #动作和随机状态构建rnn输入层          动作和前一个隐状态组合
        rnn_input = self._rnn_input_model(torch.cat([prev_action, prev_state['stoch']], dim=-1)) #at-1 st-1 经过一层网络和激活函数得到rnn input
        #gru预测 rnninput，p_state_deter（rnn输入、前一个状态隐藏层信息）:获取deter_state --RNN隐藏层输出和当前obs对应的state
        deter_state = self._cell(rnn_input, prev_state['deter'])#（at-1,st-1）+ hidden st-1  = hidden st
        mean, std = torch.chunk(self._stochastic_prior_model(deter_state), 2, dim=-1) #hidden st送到神经网络中进行信息拆解得到 st （mean、std）  ht ->st (mean,std)
        std = F.softplus(std) + 0.1 #对标准差 使用softplus激活，防止数值为负或过小
        dist = D.Independent(self._dist(mean, std), 1) #构造分布
        stoch_state = dist.rsample()   #对分布重采样，隐分布采样出st
        return dict(mean=mean, std=std, stoch=stoch_state, deter=deter_state) #意思：返回关于st分布的均值、方差，st的采样结果，hidden st的信息

class RepresentationModel(nn.Module):
    def __init__(self, obs_embed_size, action_size, stochastic_size=30,
                 deterministic_size=200, hidden_size=200, activation=nn.ELU, distribution=D.Normal):
        super().__init__()
        self._obs_embed_size = obs_embed_size
        self._action_size = action_size
        self._stoch_size = stochastic_size
        self._deter_size = deterministic_size
        self._hidden_size = hidden_size
        self._activation = activation
        self._dist = distribution
        self._stochastic_posterior_model = self._build_stochastic_model()

    def _build_stochastic_model(self):
        stochastic_model = [nn.Linear(self._deter_size + self._stoch_size + self._obs_embed_size, self._hidden_size)]
        stochastic_model += [self._activation()]
        stochastic_model += [nn.Linear(self._hidden_size, 2 * self._stoch_size)]
        return nn.Sequential(*stochastic_model)

    def initial_state(self, batch_size, **kwargs):
        return dict(
            mean=torch.zeros(batch_size, self._stoch_size, **kwargs),
            std=torch.zeros(batch_size, self._stoch_size, **kwargs),
            stoch=torch.zeros(batch_size, self._stoch_size, **kwargs),
            deter=torch.zeros(batch_size, self._deter_size, **kwargs),
        )
    #随机动作中，is init为false，也就是没有前一个状态
    def forward(self, obs_embed: torch.Tensor, prev_action: torch.Tensor, prev_state: dict, is_init=False, transition_model=None):
        # 首先要产生先验状态
        if is_init:
            prior_state = prev_state
        else:     #输入前一个动作-前一个状态
            prior_state = transition_model(prev_action, prev_state)   #返回的是先验 基于前一个状态衍生后一个状态 st（mean，std，sample，GRU（ht））
        x = torch.cat([prior_state['stoch'], prior_state['deter'], obs_embed], dim=-1)  #后验输入拼接  x=（st，hidden st，feat_vec）论文表示为st（st-1，at-1，ot）  根据先验求出当前obs下的st
        mean, std = torch.chunk(self._stochastic_posterior_model(x), 2, dim=-1) # st，ht，obs-feat-t送入到神经网络中产生后验st
        std = F.softplus(std) + 0.1
        dist = D.Independent(self._dist(mean, std), 1)
        stoch_state = dist.rsample()
        posterior_state = dict(mean=mean, std=std, stoch=stoch_state, deter=prior_state['deter'])#后验分布，即添加了ot作为参数所产出的隐藏状态。
        return prior_state, posterior_state  #都是隐藏状态，返回先验和后验st，一个是前一个状态产生的st，一个是基于前一个状态和当前feature产生的st


class VAE(nn.Module):
    def __init__(self,obs_dim,z_dim):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(obs_dim, 512)
        self.fc2_mu = nn.Linear(512, z_dim)
        self.fc2_log_std = nn.Linear(512, z_dim)
        self.fc3 = nn.Linear(z_dim, 512)
        self.fc4 = nn.Linear(512, obs_dim)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        mu = self.fc2_mu(h1)
        log_std = self.fc2_log_std(h1)
        return mu, log_std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        recon = torch.sigmoid(self.fc4(h3))
        return recon

    def reparametrize(self, mu, log_std):
        std = torch.exp(log_std)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def forward(self, x):
        mu, log_std = self.encode(x)
        z = self.reparametrize(mu, log_std)
        recon = self.decode(z)
        return recon, mu, log_std

    def loss_function(self, recon, x, mu, log_std) -> torch.Tensor:
        recon_loss = F.mse_loss(recon, x, reduction="mean")  # use "mean" may have a bad effect on gradients
        kl_loss = -0.5 * (1 + 2*log_std - mu.pow(2) - torch.exp(2*log_std))
        kl_loss = torch.sum(kl_loss)
        loss = recon_loss + kl_loss
        return loss