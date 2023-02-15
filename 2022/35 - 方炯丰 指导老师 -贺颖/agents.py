import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import numpy as np

from dense_models import *
from conv_models import *
from world_model import *
import utils

def get_mode(dist, n_samples = 100):
    sample = dist.sample_n(n_samples)
    logprob = dist.log_prob(sample)
    mode_indices = torch.argmax(logprob, dim=0)
    return sample[mode_indices]

def stack_states(rssm_states: list, dim=0):
    return dict(
        mean=torch.stack([state['mean'] for state in rssm_states], dim=dim),
        std=torch.stack([state['std'] for state in rssm_states], dim=dim),
        stoch=torch.stack([state['stoch'] for state in rssm_states], dim=dim),
        deter=torch.stack([state['deter'] for state in rssm_states], dim=dim),
    )


def flatten_state(rssm_state: dict):
    return dict(
        mean=torch.reshape(rssm_state['mean'], [-1, rssm_state['mean'].shape[-1]]),
        std=torch.reshape(rssm_state['std'], [-1, rssm_state['std'].shape[-1]]),
        stoch=torch.reshape(rssm_state['stoch'], [-1, rssm_state['stoch'].shape[-1]]),
        deter=torch.reshape(rssm_state['deter'], [-1, rssm_state['deter'].shape[-1]]),
    )


def detach_state(rssm_state: dict):
    return dict(
        mean=rssm_state['mean'].detach(),
        std=rssm_state['std'].detach(), 
        stoch=rssm_state['stoch'].detach(), 
        deter=rssm_state['deter'].detach(),
    )


def expand_state(rssm_state: dict, n : int):
    return dict(
        mean=rssm_state['mean'].expand(n, *rssm_state['mean'].shape),
        std=rssm_state['std'].expand(n, *rssm_state['std'].shape), 
        stoch=rssm_state['stoch'].expand(n, *rssm_state['stoch'].shape), 
        deter=rssm_state['deter'].expand(n, *rssm_state['deter'].shape),
    )


def get_dist(rssm_state: dict):
    return D.independent.Independent(D.Normal(rssm_state['mean'], rssm_state['std']), 1)


class Agent(nn.Module):
    def __init__(self, 
                    config=None,
                    world_lr=6e-4, 
                    policy_lr=8e-5, 
                    value_lr=8e-5, 
                    device='cuda' if torch.cuda.is_available() else 'cpu', 
                ):
        super().__init__()

        self.config = config

        self.action_dist = config['action_dist'] 
        self.use_rewards = config['use_rewards']
        self.contrastive = config['contrastive']
        
        self._action_size = config['action_size']

        self.wm = WorldModel(config)
        self.world_optim = torch.optim.Adam(utils.get_parameters([self.wm]), lr=world_lr)
            
        self.policy = ActionModel(self._action_size, 230, 200, 3, dist=self.action_dist)
        self.policy_optim = torch.optim.Adam(self.policy.parameters(), lr=policy_lr)

        self.reinforce = self.action_dist == 'one_hot'

        self.value_model = DenseModel(self.wm._feature_size, (1,), 3, 200) #价值网络
        if self.reinforce or not self.use_rewards:  #如果不使用reward
            self.value_target = DenseModel(self.wm._feature_size, (1,), 3, 200)  #目标价值网络 返回一个效用值，即每个时间步，每个batch的状态的value值
        else:
            self.value_target = self.value_model
        #优化的是value model
        self.value_optim = torch.optim.Adam(self.value_model.parameters(), lr=value_lr)

        self.grad_clip = 100.
        self.gamma = config['discount_gamma'] 
        self.device = device

        self.add_actor_entropy = config.get('actor_entropy', False)
        self.entropy_temperature = config.get('entropy_temperature', 1e-4)
        
        # Default for the moment
        self.use_rms = False
        self.rew_rms = utils.RunningMeanStd()
        self.ambiguity_rms = utils.RunningMeanStd()
        self.ambiguity_beta = 1e-3
        ##
        self.to(device)

    def update_target_network(self, tau, network, target_network):
        # Softly Update Target 
        target_value_params = target_network.named_parameters()
        value_params = network.named_parameters()

        target_value_state_dict = dict(target_value_params)
        value_state_dict = dict(value_params)

        for name in value_state_dict:
            value_state_dict[name] = tau*value_state_dict[name].clone() + \
                (1-tau)*target_value_state_dict[name].clone()
        
        target_network.load_state_dict(value_state_dict)

    def train_world(self, path_obs, path_act, path_rew, preferred_obs, path_done=None, update_target=False, get_reconstruction=False):
        loss_dict = dict()
        #time,n_path,shape
        batch_t, batch_b, img_shape = path_obs.shape[0], path_obs.shape[1], path_obs.shape[2:]
        batch_t -= 1
        #此时obs是包含初始状态的，所以需要截取
        init_obs = path_obs[0]
        next_obs = path_obs[1:]
        #吧obs进行encoder
        obs_embed = self.wm.obs_encoder(path_obs)
        #obs-feat-vec截取，与obs同理
        init_embed = obs_embed[0]
        next_embed = obs_embed[1:]
        #构造先验隐藏状态分布，针对每一个时间步进行操作
        prev_state = self.wm.prior.initial_state(batch_size=batch_b, device=self.device)
        first_embed,_ = self.wm.vae.encode(init_embed)
        prev_state["stoch"] = first_embed
        prev_state = detach_state(prev_state)

        # 时间步，隐藏状态，动作，先验分布
        prior, post = self.rollout_posterior(batch_t, next_embed, path_act, prev_state)  #返回当前next_embed的先验后验隐状态，其中按step、batch进行forward
        feat = torch.cat((post['stoch'], post['deter']), dim=-1)  #构建（后验st，ht）元组  feat 特征元组

        if self.use_rewards: #不使用reward
            reward_pred = self.wm.rew_model(feat)
            reward_loss = -torch.mean(reward_pred.log_prob(path_rew))
        else:
            reward_loss = torch.zeros(1).to(self.device)

        if self.contrastive:
            W_c = post['stoch'].reshape(batch_t * batch_b, -1) # N   没有用到
            reshaped_obs = path_obs[1:].reshape(batch_b*batch_t, *img_shape)  #没有用到
            W_c = self.wm.z_encoder(feat.detach()).reshape(batch_t * batch_b, -1)  #计算g（st）
            mean_z = self.wm.w_contrastive(next_embed.detach()).reshape(batch_t * batch_b, -1) #h（ot）  即与之对应的ot的得分
            sim_matrix = torch.mm(W_c, mean_z.T) #利用torch.mm做dot product，针对每一个st与ot做dot product，最终得到一个dot product数组，及每一个时刻一个dot product, 每一行为一个状态与其他所有状态的相似度
            labels = torch.Tensor(list(range(batch_b * batch_t))).long().to(self.device) #label数组，用于使st和ot的位置相对应
            image_loss = F.cross_entropy(sim_matrix, labels, reduction='mean') #对sim_matirx中每行进行分类，st0对应o0位置就会得分最大化，使得st和ot联系最大化
        else:
            image_pred = self.wm.obs_decoder(feat)
            image_loss = -torch.mean(image_pred.log_prob(next_obs))


        prior_dist = get_dist(prior) #构造先验st分布
        post_dist = get_dist(post)  #构造后验st分布
            
        div = torch.mean(torch.distributions.kl.kl_divergence(post_dist, prior_dist)) #计算先验st和后验st之间的kld
        kl_loss = torch.max(div, div.new_full(div.size(), self.wm.free_nats)) #计算KL，如果kl小于3则固定为3，防止后验分布塌陷，什么是后验塌陷？就是网络独立的构建分布，不根据先验和ot继续学习。

        recon, mu, log_std = self.wm.vae(obs_embed.detach())
        vae_kl_loss = self.wm.vae.loss_function(recon, obs_embed, mu, log_std)

        vae_kl_loss = vae_kl_loss.mean()*0.1
        model_loss = image_loss + reward_loss + self.wm.kl_scale * kl_loss + vae_kl_loss  #同时对得分函数，先验后验网络进行，最小化kl，最大化状态相似得分

        self.world_optim.zero_grad() #重置梯度
        model_loss.backward()   #梯度计算

        grad_norm_world = torch.nn.utils.clip_grad_norm_(utils.get_parameters([self.wm]), self.grad_clip)  #梯度裁剪  coef =  max_clip/total_clip *grad 避免梯度爆炸

        loss_dict['world_grad_norm'] = grad_norm_world #记录梯度
        
        self.world_optim.step() #梯度下降  更新feature_encoder，z_encode,w_contrastive,rnn,GRU，先验和后验的状态分离网络，即所有网络都训练到了
        loss_dict = dict(reconstruction_loss=image_loss.item(), kl_loss=kl_loss, reward_loss=reward_loss.item(), vae_loss=vae_kl_loss.item(),**loss_dict) #记录st与ot之间的loss，st和st的loss，klloss

        reconstruction_dict = dict()

        return post, loss_dict, reconstruction_dict #返回后验st，loss字典，重建字典

    def train_value(self, state_features, lambda_returns, discount_arr):
        value_pred = self.value_model(state_features.detach()) #先验状态 特征 预测状态自由能
        value_pred = D.independent.Independent(D.Normal(value_pred.mean[:-1], 1), 1) #构造自由能分布，同时去掉最后一步状态
        #要求调整value网络的分布，使得分布的可能性最大，即优化value网络
        value_loss = -torch.mean(discount_arr * value_pred.log_prob(lambda_returns.detach()) ) #从预测自由能分布中求 真实自由能的logp值 在对logp进行折扣计算求均值，要求最大化这个值，即使预测值接近真实值

        self.value_optim.zero_grad()
        value_loss.backward()
        grad_norm_value = torch.nn.utils.clip_grad_norm_(utils.get_parameters([self.value_model]), self.grad_clip)
        
        self.value_optim.step() #梯度下降

        return value_loss.item(), grad_norm_value #返回梯度项

    # 传入（时间步，策略网络，后验状态st，pre obs,obs_path）
    def train_policy_value(self, steps, policy, states, preferred_obs, obs_batch=None):
        with utils.FreezeParameters([self.wm, self.value_model]): #冻结模型参数
            states = detach_state(flatten_state(states)) #展开策略，并构造分布dict，path_obs_st的后验分布
            
            list_prior_states, act_logprobs, actions, act_entropies = self.rollout_policy(steps, policy, states)#将当前state作为初始状态进行输入，根据策略网络采样动作执行动作，预测所有状态后step步的状态

            prior_states =  stack_states(list_prior_states, dim=0) #记录本来的一步状态+先验网络预测状态

            all_prior_feat = torch.cat((prior_states['stoch'], prior_states['deter']), dim=-1) #构建先验feat，用于求出后验st

            free_energy, fe_dict = self.compute_free_energy(None, preferred_obs, prior_states, actions, obs_batch=obs_batch)  #计算action网络采样的动作进行预测的先验st和pre_obs以及与负样本之间的自由能，公式10

            loss_dict = dict(**fe_dict)

            future_value_pred = self.value_target(all_prior_feat).mean  #价值预测网咯，预测 policy网络预测的状态列表的未来自由能  #返回一个效用值，即每个时间步，每个batch的状态的value值

            if not self.use_rewards:
                if self.use_rms:
                    self.rew_rms.update(free_energy.detach().cpu().view(-1, 1).numpy())
                    free_energy = free_energy / np.sqrt(self.rew_rms.var.item() + 1e-8)

            loss_dict['free_energy'] = free_energy.detach().mean().item()   #记录自由能均值
            loss_dict['action_entropy'] = act_entropies.mean().detach().cpu().item() #记录动作 熵的均值

            discount = torch.ones_like(future_value_pred) * self.gamma #构建折扣因子函数，与自由能做乘积即可获得累计效用

            expected_free_energy = utils.lambda_dreamer_values(free_energy[:,:,None], future_value_pred, gamma=discount) #通过gae计算每个状态的未来自由能计算加权累计值，累计自由能
            loss_dict['expected_free_energy'] = expected_free_energy.detach().cpu().mean().item() #记录未来自由能的平均效用

            discount_arr = torch.cat( [discount[:1] / self.gamma, discount[:-1]], dim=0).detach()   #对第一步的动作，反gamma
            discount_arr = torch.cumprod(discount_arr, dim=0).squeeze(-1) #累乘每一步的的gamma值，每一个时间步的gamma都计算出来

            loss = torch.mean(discount_arr[:-1] * expected_free_energy[:-1].squeeze(-1))  #对所有状态的未来自由能进行折扣乘积求均值，即要求所有动作产生的状态自由能要最小化，调整策略网络

            if self.add_actor_entropy: #公式中的动作熵，鼓励探索
                loss = loss - torch.mean(discount_arr[:-1] * act_entropies * self.entropy_temperature)

            self.policy_optim.zero_grad() #置零梯度
            
            loss.backward()#回传梯度，包含动作和策略网络，首先优化策略网络
            
            grad_norm_actor = torch.nn.utils.clip_grad_norm_(utils.get_parameters([self.policy]), self.grad_clip)
            loss_dict['policy_grad_norm'] = grad_norm_actor

            self.policy_optim.step() #优化策略网络，调整网络采样的结果使动作产生的状态使平均自由能最小化，且熵最大化

        with utils.FreezeParameters([self.wm, self.policy]): #冻结其他网络
            value_loss_item, value_grad_norm = self.train_value(all_prior_feat, expected_free_energy[:-1].detach(), discount_arr[:-1])       #优化策略网络
            loss_dict['value_logprob_loss'] = value_loss_item #记录价值网络loss
            loss_dict['value_grad_norm'] = value_grad_norm #记录梯度

        return loss_dict  #返回loss字典

    def compute_free_energy(self, rew, preferred_obs, prior_states=None, actions=None, obs_batch=None): #计算自由能函数
        free_energy_dict = dict()
        free_energy = torch.zeros(1,1)
        
        if len(prior_states['stoch'].shape) == 2: #判断是否为二维状态，是的话在前面增加一维，针对随机策略
            prior_states = expand_state(prior_states,1) 
        #获取时间步长维度，多少个batch，状态维度， time batch dim
        batch_t, batch_b, state_dim = prior_states['stoch'].shape[0], prior_states['stoch'].shape[1], prior_states['stoch'].shape[2]
        # Contrastive AIF
        if self.contrastive:
            feat = torch.cat((prior_states['stoch'], prior_states['deter']), dim=-1) #拼接（st,ht）形成隐藏feat结构
            pref_embed = self.wm.obs_encoder(preferred_obs).view(1, self.wm.obs_encoder.embed_size) #抽取pre_obs特征，feat
            #为的是减少st和ot之间的互信息，就像从st推断出ot
            W_c = self.wm.z_encoder(feat).reshape(batch_t * batch_b, -1)   #对st进行计算信息得分 h(st)
            pref_z = self.wm.w_contrastive(pref_embed).reshape(1, 200) #对pre-obs-vec进行计算信息得分   g（ot）
            #计算出每步的所有状态与pre_obs的得分
            pos_loss = torch.mm(W_c, pref_z.T).view(batch_t, batch_b) / W_c.shape[-1]  #pos_loss，代表论文10中第一项，该项表示了st，pre_ot之间的共享信息量 .
            if obs_batch is None:
                vae_o = self.wm.vae.decode(prior_states['stoch'])
                recon, mu, log_std = self.wm.vae(vae_o)
                vae_kl_loss= self.wm.vae.loss_function(recon,vae_o,mu,log_std)
                vae_kl_loss = vae_kl_loss*0.1
                free_energy = -pos_loss - vae_kl_loss
                free_energy_dict['-vae_loss'] = vae_kl_loss.detach().mean().item()
            free_energy_dict['-pos_loss'] = pos_loss.detach().mean().item()


            # Only to allow computation along the episode
            if obs_batch is not None:  #在训练策略网络过程中使用
                next_embed = self.wm.obs_encoder(obs_batch).view(-1, self.wm.obs_encoder.embed_size) #对epi中采样出来的所有样本进行encoder
                mean_z = self.wm.w_contrastive(next_embed).reshape(-1, 200) #计算所有特征的互信息得分
                mean_z = torch.cat([pref_z, mean_z], dim=0) #将ot的互信息得分和pre_o的互信息得分进行拼接
                sim_matrix = torch.mm(W_c, mean_z.T)  / W_c.shape[-1] #计算st_ot之间的互信息得分，对应公式10的第二项，计算的是st对所有ot的信息得分
                neg_loss = torch.logsumexp(sim_matrix, dim=1).view(batch_t, batch_b) - np.log(mean_z.shape[0])  #计算的是每个st对应所有ot的累计信息熵的均值，即负样本，要求最小化
                free_energy = -pos_loss + neg_loss
                free_energy_dict['neg_loss'] = neg_loss.detach().mean().item()


        return free_energy, free_energy_dict


    def rollout_policy(self, steps, policy, prev_state):
        priors = [prev_state] #所有的状态列表，对所有状态进行act
        act_logprobs = []
        actions = []
        act_entropies = []
        state = prev_state
        for t in range(steps): #重复预测所有状态step步的先验st
            # Act
            feat = torch.cat((state['stoch'], state['deter']), dim=-1) #后验隐藏状态
            feat = feat.detach() 
            act_dist = policy(feat) #根据后验st构成的feat生成动作分布

            if self.reinforce:
                act = act_dist.sample()#对每个状态采样出动作
            else:
                act = act_dist.rsample() 
            
            actions.append(act)  #添加当前步的动作
            act_entropies.append(act_dist.entropy())#计算当前动作在分布中的熵
            act_logprobs.append(act_dist.log_prob(act))#计算当前动作的log 概率

            # Imagine
            state = self.wm.prior(actions[t], state) #对当前所有路径中的状态进行先验st计算，得到所有状态的先验st+1
            priors.append(state) #加入先验状态
        #收集step个步长的隐藏状态数据，动作样本st+1 ---> st+1+steps
        actions = torch.stack(actions, dim=0) # shape (T, B, action_dim)  #返回动作
        act_logprobs = torch.stack(act_logprobs, dim=0) # shape (T, B, action_dim) #返回动作log 概率
        act_entropies = torch.stack(act_entropies, dim=0) # shape (T, B, action_dim) #返回动过熵

        return priors, act_logprobs, actions, act_entropies  ##返回先验状态队列 t，b，size，动作序列 t-1，b，size等信息
    #动作生成，预测步长，策略网络，前一个状态
    def rollout_policies(self, steps, policies, prev_state, take_mean_action=False):
        priors = []
        actions = []
        state = prev_state
        for t in range(steps): #进行一步预测
            # Act
            feat = torch.cat((state['stoch'], state['deter']), dim=-1) #隐状态 特征
            feat = feat.detach()
            act = torch.stack([ p(f).sample() for f, p in zip(feat, policies)], dim=0) #使用策略网络对当前状态进行采样，采样出动作
            actions.append(act)
            # Imagine
            state = self.wm.prior(actions[t], state) #使用先验网络预测下一步的state
            priors.append(state)
        
        all_prior_states =  stack_states(priors, dim=0) #记录下一个状态列表
        actions = torch.stack(actions, dim=1) # shape: (B,T,action_dim) #记录每一个action
        return all_prior_states, actions

    # 输入当前状态、reward、动作、模型产生的前一个状态--》初始为none
    def step(self, image_obs, rew, act, prev_state):

        with torch.no_grad(): #ot+1
            image_embed = self.wm.obs_encoder(image_obs)  #通过CNN卷积特征提取进行对输入图片进行encoder,feature  该步提取特征向量
            if prev_state is None:
                #这里改为使用VAE进行编码生成隐藏空间
                prev_state = self.wm.posterior.initial_state(batch_size=1, device=self.device)  #首次初始化一个全零后验状态字典（mean、std、stoch、deter）个人认为代表隐藏状态
            #将1，encoder后的图片，动作，初始化的前一个状态输入，得到represent模型的后验分布，先验和后验输出都用于参数化表示随机状态的高斯多元分布，我们使用重参数化技巧[27]从中抽样
            _, post = self.rollout_posterior(1, image_embed, act, prev_state)  #将步长、feature 、动作、前一个隐状态输入到后验网络中  再随机网络中得到一个后验st
        return flatten_state(post) #计算出整个路径的st

    def eval_obs(self, image_obs, rew, preferred_obs, prev_state):
        with torch.no_grad():
            preferences = self.compute_free_energy(None, preferred_obs, prior_states=prev_state)[0]   #（不给rew，给出pre_obs，st作为先验状态）
        return preferences
    #在策略收集阶段，预测一步策略得出动作，更具前一个隐状态得出当前隐状态，然后计算隐状态和pre-obs的自由能
    def policy_distribution(self, steps, policies, preferred_obs, prev_state=None, eval_mode=False):
        with torch.no_grad():
            n_policies = len(policies) #网络数量
            if prev_state is None:  #前一个状态为none，创建新的先验隐状态，batch size为策略网络数量，后续可以同时训练几个网络？
                prev_state = self.wm.posterior.initial_state(batch_size=n_policies, device=self.device)

            prev_state = flatten_state(prev_state)
            all_prior_states, actions = self.rollout_policies(steps, policies, prev_state, take_mean_action=eval_mode) #根据前一个状态，通过策略网络得出动作，再根据动作和state输入到先验网络中得出st+1
            all_prior_feat = torch.cat((all_prior_states['stoch'], all_prior_states['deter']), dim=-1) #构建隐藏状态 feat

            #计算动作预测后预测状态和偏好状态之间的互信息loss，即简化自由能
            preference_loss = torch.mean(self.compute_free_energy(None, preferred_obs, prior_states=all_prior_states)[0], dim=0)

            free_energy = preference_loss
            #计算策略选择概率，选出自由能得分最大的策略，得分为负数，需要转为正数进行softmax
            policy_logits = F.softmax(-free_energy.detach(), dim=0)
            
            policy_distr = D.Categorical(policy_logits) #将概率数组转化为动作选择离散分布
            expected_loss = torch.sum(free_energy.detach() * policy_logits) #计算期望得分
            return policy_distr, actions.detach(), dict(policy_expected_loss=expected_loss.detach().cpu().item()) #返回策略概率分布，动作，期望自由能

    def rollout_posterior(self, steps: int, obs_embed: torch.Tensor, action: torch.Tensor, prev_state: dict):
        #随机策略阶段 将1，encoder后的图片特征，动作，初始化的状态字典
        #策略训练阶段  有time step，需要对整个数组进行预测
        priors = []
        posteriors = []
        prev_state = detach_state(prev_state)
        obs_embed = obs_embed.detach()
        for t in range(steps): #再随机策略时 只做一步动作
            #调用后验模型，产生先验状态和后验状态，再随机模型中输入下一个状态的特征、做的动作，前一个隐状态
            prior_state, posterior_state = self.wm.posterior(obs_embed[t], action[t], prev_state, transition_model=self.wm.prior)  #获得先验st 和后验st
            priors.append(prior_state) #先验st列表
            posteriors.append(posterior_state)  #后验st列表
            prev_state = posterior_state #后验产生的st作为前一个状态送入先验网络产生st，ht
        prior = stack_states(priors, dim=0) #list转字典
        post = stack_states(posteriors, dim=0)
        return prior, post #最终得到（s1，sT）的先验st和后验st

    def rollout_prior(self, steps: int, action: torch.Tensor, prev_state: dict):
        priors = []
        state = prev_state
        for t in range(steps):
            state = self.wm.prior(action[t], state)
            priors.append(state)
        return stack_states(priors, dim=0)
