import os
from tqdm import tqdm
import warnings
import torch
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from tensorboard_logger import Logger as TbLogger
import numpy as np
import random

from utils import clip_grad_norms, rotate_tensor
from neural_network.actor_network import Actor
from neural_network.critic_network import Critic
from utils import torch_load_cpu, get_inner_model, move_to, move_to_cuda
from utils.logger import log_to_tb_train
from training_algorithm.utils import validate

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []  
        self.obj = []
        self.action_record = []
        
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.obj[:]
        del self.action_record[:]

class PPO:
    def __init__(self, problem_name, size, parameters):
        
        # figure out the options
        self.parameters = parameters
        
        # figure out the actor
        self.actor = Actor(
            problem_name = problem_name,
            embedding_dim = parameters.embedding_dim,
            hidden_dim = parameters.hidden_dim,
            n_heads_actor = parameters.actor_head_num,
            n_layers = parameters.n_encode_layers,
            normalization = parameters.normalization,
            v_range = parameters.v_range,
            seq_length = size + 1
        )
        
        if not parameters.eval_only:
        
            # figure out the critic
            self.critic = Critic(
                    problem_name = problem_name,
                    embedding_dim = parameters.embedding_dim,
                    hidden_dim = parameters.hidden_dim,
                    n_heads = parameters.critic_head_num,
                    n_layers = parameters.n_encode_layers,
                    normalization = parameters.normalization
                )
        
            # figure out the optimizer
            self.optimizer = torch.optim.Adam(
            [{'params':  self.actor.parameters(), 'lr': parameters.lr_model}] + 
            [{'params':  self.critic.parameters(), 'lr': parameters.lr_critic}])
            
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, parameters.lr_decay, last_epoch=-1,)
                
        
        print(f'Distributed: {parameters.distributed}')
        if parameters.use_cuda and not parameters.distributed:
            
            self.actor.to(parameters.device)
            if not parameters.eval_only: self.critic.to(parameters.device)
            
            if torch.cuda.device_count() > 1:
                self.actor = torch.nn.DataParallel(self.actor)
                if not parameters.eval_only: self.critic = torch.nn.DataParallel(self.critic)
                
    
    def load(self, path):
        
        assert path is not None
        data = torch_load_cpu(path)
        # 加载动作网络的数据
        actor_model = get_inner_model(self.actor)
        actor_model.load_state_dict({**actor_model.state_dict(), **data.get('actor', {})})
        
        if not self.parameters.eval_only:
            # 加载评价网络的数据
            critic_model = get_inner_model(self.critic)
            critic_model.load_state_dict({**critic_model.state_dict(), **data.get('critic', {})})
            # 加载数据优化器
            self.optimizer.load_state_dict(data['optimizer'])
            # 为torch和cuda加载数据
            torch.set_rng_state(data['rng_state'])
            if self.parameters.use_cuda:
                torch.cuda.set_rng_state_all(data['cuda_rng_state'])
        # 输出
        print(' [*] getting data from {}'.format(path))
        
    
    def save(self, epoch):
        print('保存模型和状态')
        torch.save(
            {
                'actor': get_inner_model(self.actor).state_dict(),
                'critic': get_inner_model(self.critic).state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'rng_state': torch.get_rng_state(),
                'cuda_rng_state': torch.cuda.get_rng_state_all(),
            },
            os.path.join(self.parameters.save_dir, 'epoch-{}.pt'.format(epoch))
        )
    
    
    def eval(self):
        torch.set_grad_enabled(False)
        self.actor.eval()
        if not self.parameters.eval_only: self.critic.eval()
        
    def train(self):
        torch.set_grad_enabled(True)
        self.actor.train()
        if not self.parameters.eval_only: self.critic.train()
    
    def rollout(self, problem, val_m, batch, do_sample = False, show_bar = False):
        batch = move_to(batch, self.parameters.device) # batch_size, graph_size, 2     
        bs, gs, dim = batch['coordinates'].size()
        batch['coordinates'] = batch['coordinates'].unsqueeze(1).repeat(1,val_m,1,1)
        augments = ['Rotate', 'Flip_x-y', 'Flip_x_cor', 'Flip_y_cor']
        if val_m > 1:
            for i in range(val_m):
                random.shuffle(augments)
                id_ = torch.rand(4)
                for aug in augments:
                    if aug == 'Rotate':
                        batch['coordinates'][:,i] = rotate_tensor(batch['coordinates'][:,i], int(id_[0] * 4 + 1) * 90)
                    elif aug == 'Flip_x-y':
                        if int(id_[1] * 2 + 1) == 1:
                             data = batch['coordinates'][:,i].clone()
                             batch['coordinates'][:,i,:,0] = data[:,:,1]
                             batch['coordinates'][:,i,:,1] = data[:,:,0]
                    elif aug == 'Flip_x_cor':
                        if int(id_[2] * 2 + 1) == 1:
                             batch['coordinates'][:,i,:,0] = 1 - batch['coordinates'][:,i,:,0]
                    elif aug == 'Flip_y_cor':
                        if int(id_[3] * 2 + 1) == 1:
                             batch['coordinates'][:,i,:,1] = 1 - batch['coordinates'][:,i,:,1]
                             
        batch['coordinates'] =  batch['coordinates'].view(-1, gs, dim)
        solutions = move_to(problem.get_initial_solutions(batch, val_m), self.parameters.device).long()
        
        obj = problem.cost(batch, solutions)
        
        obj_history = [torch.cat((obj[:,None],obj[:,None]),-1)]
        reward = []

        feature2batch = problem.input_feature_encoding(batch)

        exchange = None
        action_record = [torch.zeros((feature2batch.size(0), problem.size//2)) for i in range(problem.size//2)]
        

        for t in tqdm(range(self.parameters.T_max), disable = self.parameters.no_progress_bar or not show_bar, desc = 'rollout', bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}'):       
            
            # pass through model
            exchange = self.actor(problem,
                                  feature2batch,
                                  solutions,
                                  exchange,
                                  action_record,
                                  do_sample = do_sample)[0]
            
            # new sol
            solutions, rewards, obj, action_record = problem.step(batch, solutions, exchange, obj, action_record)

            # record informations
            reward.append(rewards)
            obj_history.append(obj)

        out = (obj[:,-1].reshape(bs, val_m).min(1)[0], # batch_size, 1
               torch.stack(obj_history,1)[:,:,0].view(bs, val_m, -1).min(1)[0],  # batch_size, T
               torch.stack(obj_history,1)[:,:,-1].view(bs, val_m, -1).min(1)[0],  # batch_size, T
               torch.stack(reward,1).view(bs, val_m, -1).max(1)[0], # batch_size, T
               )

        return out


    def inference(self, problem, ver_data, logger):
        if self.parameters.distributed:
            mp.spawn(validate, nprocs=self.parameters.world_size, args=(problem, self, ver_data, logger, True))
        else:
            validate(0, problem, self, ver_data, logger, distributed = False)

    def training(self, problem, ver_data, logger):
        if self.parameters.distributed:
            mp.spawn(train, nprocs=self.parameters.world_size, args=(problem, self, ver_data, logger))
        else:
            train(0, problem, self, ver_data, logger)

def train(rank, problem, model, ver_data, logger):

    parameters = model.parameters
    #过滤提示消息
    warnings.filterwarnings("ignore")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)

    if parameters.distributed:
        machine = torch.device("cuda", rank)
        torch.distributed.init_process_group(backend='nccl', world_size=parameters.world_size, rank = rank)
        torch.cuda.set_device(rank)
        model.actor.to(machine)
        model.critic.to(machine)
        for state in model.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(machine)

        if torch.cuda.device_count() > 1:
            model.actor = torch.nn.parallel.DistributedDataParallel(model.actor,
                                                                   device_ids=[rank])
            if not parameters.eval_only: model.critic = torch.nn.parallel.DistributedDataParallel(model.critic,
                                                                   device_ids=[rank])
        if not parameters.no_tb and rank == 0:
            logger = TbLogger(os.path.join(parameters.log_dir, "{}_{}".format(parameters.problem,
                                                          parameters.graph_size), parameters.run_name))
    else:
        for state in model.optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(parameters.device)

    if parameters.distributed: dist.barrier()

    # 开始训练循环过程
    for epoch in range(parameters.epoch_start, parameters.epoch_end):

        model.lr_scheduler.step(epoch)

        # 训练模式
        if rank == 0:
            print('\n\n')
            print("|",format(f" 训练 epoch {epoch} ","*^60"),"|")
            print("Training with actor lr={:.3e} critic lr={:.3e} for run {}".format(model.optimizer.param_groups[0]['lr'],
                                                                                 model.optimizer.param_groups[1]['lr'], parameters.run_name) , flush=True)
        # 预训练数据
        training_data = problem.make_dataset(size=parameters.graph_size, num_samples=parameters.epoch_size)
        if parameters.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(training_data, shuffle=False)
            training_dataloader = DataLoader(training_data, batch_size=parameters.batch_size // parameters.world_size, shuffle=False,
                                            num_workers=0,
                                            pin_memory=True,
                                            sampler=sampler)
        else:
            training_dataloader = DataLoader(training_data, batch_size=parameters.batch_size, shuffle=False,
                                                       num_workers=0,
                                                       pin_memory=True)

        # 开始训练
        step = epoch * (parameters.epoch_size // parameters.batch_size)
        processbar = tqdm(total = (parameters.K_epochs) * (parameters.epoch_size // parameters.batch_size) * (parameters.T_train // parameters.n_step) ,
                    disable = parameters.no_progress_bar or rank!=0, desc = 'training',
                    bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        for batch_id, batch in enumerate(training_dataloader):
            train_batch(rank,
                        problem,
                        model,
                        epoch,
                        step,
                        batch,
                        logger,
                        parameters,
                        processbar,
                        )
            step += 1
        processbar.close()

        # 保留一个epoch后的新模型
        if rank == 0 and not parameters.distributed:
            if not parameters.no_saving and (( parameters.checkpoint_epochs != 0 and epoch % parameters.checkpoint_epochs == 0) or \
                        epoch == parameters.epoch_end - 1): model.save(epoch)
        elif parameters.distributed and rank == 1:
            if not parameters.no_saving and (( parameters.checkpoint_epochs != 0 and epoch % parameters.checkpoint_epochs == 0) or \
                        epoch == parameters.epoch_end - 1): model.save(epoch)


        # 验证新模型
        if rank == 0 and not parameters.distributed: validate(rank, problem, model, ver_data, logger, _id = epoch)
        if rank == 0 and parameters.distributed: validate(rank, problem, model, ver_data, logger, _id = epoch)

        if parameters.distributed: dist.barrier()


def train_batch(
        rank,
        problem,
        model,
        epoch,
        step,
        batch,
        logger,
        parameters,
        processbar,
        ):

    # 启动
    model.train()
    memory_content = Memory()

    # 预处理
    batch = move_to_cuda(batch, rank) if parameters.distributed else move_to(batch, parameters.device)# batch_size, graph_size, 2
    feature2batch = problem.input_feature_encoding(batch).cuda() if parameters.distributed \
                        else move_to(problem.input_feature_encoding(batch), parameters.device)
    batch_size = feature2batch.size(0)
    exchange = move_to_cuda(torch.tensor([-1,-1,-1]).repeat(batch_size,1), rank) if parameters.distributed \
                        else move_to(torch.tensor([-1,-1,-1]).repeat(batch_size,1), parameters.device)
    action_record = [torch.zeros((feature2batch.size(0), problem.size//2)) for i in range(problem.size)]

    # 初始化
    sol = move_to_cuda(problem.get_initial_solutions(batch),rank) if parameters.distributed \
                        else move_to(problem.get_initial_solutions(batch), parameters.device)
    fitness = problem.cost(batch, sol)

    # 热启动
    if parameters.warm_up:
        model.eval()

        for w in range(int(epoch // parameters.warm_up)):

            # 模型输出
            exchange = model.actor( problem,
                                    feature2batch,
                                    sol,
                                    exchange,
                                    action_record,
                                    do_sample = True)[0]

            # 状态过渡
            sol, rewards, fitness, action_record = problem.step(batch, sol, exchange, fitness, action_record)

        fitness = problem.cost(batch, sol)

        model.train()

    # 参数训练
    gamma = parameters.gamma
    n_step = parameters.n_step
    T = parameters.T_train
    K_epochs = parameters.K_epochs
    eps_clip = parameters.eps_clip
    t = 0
    initial_cost = fitness

    # 轨迹抽样
    while t < T:

        t_s = t
        memory_content.actions.append(exchange)

        total_cost = 0

        # 第一步
        entropy = []
        bl_val_detached = []
        bl_val = []

        while t - t_s < n_step and not (t == T):


            memory_content.states.append(sol)
            memory_content.action_record.append(action_record.copy())

            # 模型的输出
            exchange, log_lh, _to_critic, entro_p  = model.actor(problem,
                                                                 feature2batch,
                                                                 sol,
                                                                 exchange,
                                                                 action_record,
                                                                 do_sample = True,
                                                                 require_entropy = True,# take same action
                                                                 to_critic = True)

            memory_content.actions.append(exchange)
            memory_content.logprobs.append(log_lh)
            memory_content.obj.append(fitness.view(fitness.size(0), -1)[:,-1].unsqueeze(-1))


            entropy.append(entro_p.detach().cpu())

            baseline_val_detached, baseline_val = model.critic(_to_critic, fitness.view(fitness.size(0), -1)[:,-1].unsqueeze(-1))

            bl_val_detached.append(baseline_val_detached)
            bl_val.append(baseline_val)

            # 状态过渡
            sol, rewards, fitness, action_record = problem.step(batch, sol, exchange, fitness, action_record)
            memory_content.rewards.append(rewards)

            # 信息存储
            total_cost = total_cost + fitness[:,-1]

            # 下一步
            t = t + 1


        # 保存信息
        time = t - t_s
        total_cost = total_cost / time

        # 开始更新        =======================

        # 将列表转换为张量
        all_actions = torch.stack(memory_content.actions)
        old_states = torch.stack(memory_content.states).detach().view(time, batch_size, -1)
        old_actions = all_actions[1:].view(time, -1, 3)
        old_logprobs = torch.stack(memory_content.logprobs).detach().view(-1)
        old_exchange = all_actions[:-1].view(time, -1, 3)
        old_action_records = memory_content.action_record

        old_obj = torch.stack(memory_content.obj)

        # 针对K个小epoch优化ppo策略:
        old_value = None

        for _k in range(K_epochs):

            if _k == 0:
                logprobs = memory_content.logprobs

            else:
                # 评估旧的动作和cost
                logprobs = []
                entropy = []
                bl_val_detached = []
                bl_val = []

                for tt in range(time):
                    # 获取新的动作
                    _, log_p, _to_critic, entro_p = model.actor(problem,
                                                                feature2batch,
                                                                old_states[tt],
                                                                old_exchange[tt],
                                                                old_action_records[tt],
                                                                fixed_action = old_actions[tt],
                                                                require_entropy = True,# take same action
                                                                to_critic = True)

                    logprobs.append(log_p)
                    entropy.append(entro_p.detach().cpu())

                    baseline_val_detached, baseline_val = model.critic(_to_critic, old_obj[tt])

                    bl_val_detached.append(baseline_val_detached)
                    bl_val.append(baseline_val)

            logprobs = torch.stack(logprobs).view(-1)
            entropy = torch.stack(entropy).view(-1)
            bl_val_detached = torch.stack(bl_val_detached).view(-1)
            bl_val = torch.stack(bl_val).view(-1)


            # 获取评价网络的目标值
            Reward = []
            reward_reversed = memory_content.rewards[::-1]

            # 评估返回
            Re = model.critic(model.actor(problem,feature2batch,sol,exchange,action_record,only_critic = True), fitness.view(fitness.size(0), -1)[:,-1].unsqueeze(-1))[0]
            for r in range(len(reward_reversed)):
                Re = Re * gamma + reward_reversed[r]
                Reward.append(Re)


            # 锁定目标:
            Reward = torch.stack(Reward[::-1], 0)
            Reward = Reward.view(-1)

            # 算出比率
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # 找到代理损失:
            advantages = Reward - bl_val_detached

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-eps_clip, 1+eps_clip) * advantages
            reinforce_learning_loss = -torch.min(surr1, surr2).mean()

            # 定义基准损耗
            if old_value is None:
                baseline_loss = ((bl_val - Reward) ** 2).mean()
                old_value = bl_val.detach()
            else:
                vpredclipped = old_value + torch.clamp(bl_val - old_value, - eps_clip, eps_clip)
                v_max = torch.max(((bl_val - Reward) ** 2), ((vpredclipped - Reward) ** 2))
                baseline_loss = v_max.mean()

            # 检查K-L散度
            approx_kl_divergence = (.5 * (old_logprobs.detach() - logprobs) ** 2).mean().detach()
            approx_kl_divergence[torch.isinf(approx_kl_divergence)] = 0

            # 计算lost
            loss = baseline_loss + reinforce_learning_loss #- 1e-5 * entropy.mean()

            # 更新梯度步骤
            model.optimizer.zero_grad()
            loss.backward()

            # 裁剪梯度范数并获取(裁剪)梯度范数用于日志
            current_step = int(step * T / n_step * K_epochs + (t-1)//n_step * K_epochs  + _k)
            grad_norms = clip_grad_norms(model.optimizer.param_groups, parameters.max_grad_norm)

            # 执行梯度下降
            model.optimizer.step()

            # 记录到tensorboard
            if(not parameters.no_tb) and rank == 0:
                if (current_step + 1) % int(parameters.log_step) == 0:
                    log_to_tb_train(logger, model, Reward, ratios, bl_val_detached, total_cost, grad_norms, memory_content.rewards, entropy, approx_kl_divergence,
                       reinforce_learning_loss, baseline_loss, logprobs, initial_cost, parameters.show_figs, current_step + 1)
                    
            if rank == 0: processbar.update(1)     
        
        
        # 更新结束
        memory_content.clear_memory()

    
        
    
