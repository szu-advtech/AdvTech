import torch
import numpy as np
import time
from tensorboardX import SummaryWriter
import argparse
import os
import warnings
import ruamel.yaml as yaml
import pathlib
import sys
from pathlib import Path
import datetime
import json

import agents
import utils
import wrappers
from agents import Agent

warnings.filterwarnings('ignore', '.*box bound precision lowered.*')


# os.environ['MUJOCO_GL'] = 'egl'

def visualize_current_obs(obs):
    import matplotlib.pyplot as plt
    plt.imshow(obs.cpu().permute(1, 2, 0) + 0.5)
    plt.show()


@utils.retry
def save_model(model, save_dir):
    torch.save(model, save_dir)


def collect_random_episode(env, model, preferred_obs, episode_store, returns, free_energies, config):
    with torch.no_grad():
        device = config['device']
        episode = dict(obs=[], act=[], rew=[], free_energy=[], done=[])
        timestep = env.reset()  # timestep-图片矩阵
        obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])  # 标准化图片矩阵
        episode['obs'].append(obs.cpu())  # 先加入了初始状态
        first_embed = model.wm.obs_encoder(obs)
        _,mu,logstd = model.wm.vae(first_embed)
        z = model.wm.vae.reparametrize(mu,logstd)
        dic = model.wm.posterior.initial_state(1,device=device)
        dic["stoch"] = z.reshape(1,-1)
        prev_state = agents.detach_state(dic)
        done = False
        cur_return = 0
        cur_free_energy = 0
        while not done:  # 重复随机采样  注意  先加入了初始状态，后续需要注意
            action = env.action_space.sample()  # 随机采样动作

            timestep, rew, done, info = env.step(action)  # 执行动作，注意repeat  timelimit限制step次数超出1000则done
            obs = wrappers.get_scaled_obs(timestep, device=device,
                                          is_minigrid='minigrid' in config['suite'])  # 标准化下一个状态

            rew_tensor = torch.Tensor([rew]).to(device)
            done_tensor = torch.Tensor([done]).to(device)

            current_obs = obs.expand(1, 1, *obs.shape)  # 当前动作完成后的状态 加入世界模型进行forward 前面增加两个维度
            # 输入当前状态、reward、动作、模型产生的前一个状态--初始状况下为none，运行一步后得到隐藏状态st作为pre state，当前pre state作为下一个step的输入  ，后验状态作为pre state 而不是使用先验
            prev_state = model.step(current_obs.to(device), torch.Tensor([rew]).reshape(1, 1, 1).to(device),
                                    torch.from_numpy(action).to(device).view(1, 1, *action.shape).to(device),
                                    prev_state)
            # 为了计算当前动作的自由能，上面用于计算，输入当前obs的隐藏状态 （ot，rew，pref_obs，后验st）
            free_energy = model.eval_obs(obs, rew_tensor, preferred_obs, prev_state)  # 计算当前st和pre-obs之间的互信息得分
            # 每执行一次动作，即将所有属性添加到 episode队列中，其中obs队列添加了初始状态，长度为t+1
            episode['obs'].append(obs.cpu())
            episode['act'].append(torch.from_numpy(action))
            episode['rew'].append(rew_tensor.cpu())
            episode['done'].append(done_tensor.cpu())
            episode['free_energy'].append(free_energy.cpu())

            cur_return += rew
            # 累积自由能
            cur_free_energy += free_energy.cpu().numpy()
        # 执行完一个episode，写入全局存储空间、free——energies，1个epi
        episode_store.add(np.stack(episode['obs'], axis=0), np.stack(episode['act'], axis=0),
                          np.stack(episode['rew'], axis=0), np.stack(episode['free_energy'], axis=0),
                          np.stack(episode['done'], axis=0))
        returns.append(cur_return)  # 单个episode的累计值
        free_energies.append(cur_free_energy)  # 单个episode的累计值


def collect_eval_episode(env, model, preferred_obs, config, policy_lookahead=1):
    with torch.no_grad():
        device = config['device']
        # Reset
        timestep = env.reset()
        obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])
        prev_state = None
        cur_return = 0
        cur_free_energy = 0
        policy_dict = dict()
        done = False
        while not done:
            policy_distr, policy_actions, future_loss_dict = model.policy_distribution(policy_lookahead, [model.policy],
                                                                                       preferred_obs, prev_state,
                                                                                       eval_mode=True)

            for k in future_loss_dict:
                if k in policy_dict:
                    policy_dict[k] += future_loss_dict[k]
                else:
                    policy_dict[k] = future_loss_dict[k]

            # Action
            for policy_step in range(policy_lookahead):
                action_index = policy_distr.sample()
                action = policy_actions[action_index][policy_step].cpu()

                timestep, rew, done, info = env.step(action.numpy())
                obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])

                rew_tensor = torch.Tensor([rew]).to(device)
                # Minigrid only: to simplify reward prediction, as the agent has no knowledge of time
                if 'minigrid' in config['suite'] and rew_tensor > 0:
                    rew_tensor = torch.ones_like(rew_tensor)
                #
                done_tensor = torch.Tensor([done]).to(device)
                current_obs = obs.expand(1, 1, *obs.shape)
                prev_state = model.step(current_obs.to(device), torch.Tensor([rew]).reshape(1, 1, 1).to(device),
                                        action.view(1, 1, *action.shape).to(device), prev_state)

                free_energy = model.eval_obs(obs, rew_tensor, preferred_obs, prev_state)

                cur_return += rew
                cur_free_energy += free_energy.cpu().numpy()

                if done:
                    break
    return cur_return, cur_free_energy


def main(config):
    # Setup logs
    if config['seed'] is not None:
        seed = int(config['seed'])
        torch.manual_seed(seed)
        np.random.seed(seed)
    seed_str = str(datetime.datetime.now().timestamp()) if config['seed'] is None else 'seed_' + str(seed)  # 随机种子设置
    logdir = Path(config['logdir']) / config['suite'] / config['task'] / '_'.join(config['config']) / config[
        'algo'] / seed_str  # 创建tenserboard存储路径

    if not os.path.isdir(logdir):
        os.makedirs(logdir)

    writer = SummaryWriter(logdir=str(logdir))  # 设置日志写入器
    with open(str(logdir / 'config.json'), 'w') as fp:
        json.dump(config, fp, indent=4)

    # Options
    device = 'cuda:0' if torch.cuda.is_available() and not config['disable_cuda'] else 'cpu'
    config['device'] = device

    # Create env
    env = wrappers.make_env(suite=config['suite'], task_name=config['task'],
                            grid_size=config['grid_size'])  # 根据传入任务生成不同环境

    config['action_size'] = env.action_space.shape[0]  # config记录动作 维度
    action_repeat = env._action_repeat

    # Setup model
    model = Agent(device=device, config=config)

    preferred_obs = wrappers.get_scaled_obs(dict(image=torch.load(f'preferred_states/{config["task"]}.pt')), device,
                                                is_minigrid='minigrid' in config['suite'])
    policy_lookahead = 1
    expl_amount = config['expl_amount']

    # Setup training  定义参数
    total_steps = int(config['total_steps'])
    # 最大epi数
    max_episodes = config['max_episodes'] if config['max_episodes'] is not None else int(sys.maxsize)
    episode_store = utils.EpisodeStore()  # 构建全局epi存储空间
    balance_ends = config['action_dist'] == 'one_hot'

    current_step = 0
    tot_episodes = 0
    century = 0

    # Populate episode store  用随机策略填充episode store的数据
    # random_init_episodes = config['random_init_episodes']
    random_init_episodes = 10
    print(f"Collecting {random_init_episodes} episodes for init...")
    init_returns = []  # 初始化回报数组
    init_free_energies = []  # 初始化自由能数组
    while len(
            init_returns) < random_init_episodes:  # 随机策略收集episode的数据，需要收集够所需的episode，每次收集都需要对环境初始化，并不对网络进行训练，产生世界模型初始分布
        collect_random_episode(env, model, preferred_obs, episode_store, init_returns, init_free_energies, config)
    print("Random collection completed...", ' Return: ', np.round(np.mean(init_returns), 2))
    writer.add_scalar('environment/return', np.mean(init_returns), global_step=current_step)
    writer.add_scalar('environment/free_energy', np.mean(init_free_energies), global_step=current_step)
    # 现在episode中拥有n个随机策略的ep
    # Reset
    episode = dict(obs=[], act=[], rew=[], free_energy=[], done=[])  # 重置数据，清楚每次ep缓存
    timestep = env.reset()  # 重置环境
    if config['render_every'] > 0 and tot_episodes % config['render_every'] == 0:  # 输出图像设置
        env.render()
    obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])  # 返回标准化图片矩阵
    episode['obs'].append(obs.cpu())  # 添加初始状态obs队列   该队列长度长度为n+1
    first_embed = model.wm.obs_encoder(obs)
    first_embed, _ = model.wm.vae.encode(first_embed)
    dic = model.wm.posterior.initial_state(1,device=device)
    dic["stoch"] = first_embed.reshape(1, -1)
    prev_state = agents.detach_state(dic)
    cur_return = 0
    policy_dict = dict()  # 策略为空
    done = True

    while current_step < total_steps:  # 策略动作采取的次数限制
        if tot_episodes >= max_episodes:  # epi次数限制
            print('Maximum Episodes reached!')
            break

        if done and episode_store.n_episodes > 0:  # 如果一段epi搜集完成，并拥有数据，则对模型和策略进行训练
            print(f'E: {tot_episodes} | Updating model and policies...')
            n_epochs = config['n_epochs']  # 网络训练循环次数
            n_paths = config['n_paths']  # 截取路径条数，从epi内存中抽取30条epi
            n_steps = config['n_steps']  # 截取步长，从路径中抽取的步长
            horizon = config['horizon']  # 预测未来的步数
            main_loss_dict = dict()
            main_policy_loss_dict = dict()
            for i in range(n_epochs):  # 训练epoch
                log_time = i == n_epochs - 1  # 最后一轮记录，为true 其他时间为false
                log_images = (config['recon_every'] > 0) and (
                            tot_episodes % config['recon_every'] == 0) and log_time  # 是否记录状态
                # 返回多条episode的连续属性【num-ep，len】
                path_obs, path_act, path_rew, path_done = episode_store.sample_paths(n_paths, n_steps,
                                                                                     balance_ends=balance_ends)
                # 判断是否进行更新标识符，每一100轮更新一次
                update_target = i + 1 % 100 == 0
                # 传入（path_info,...,...）,pre_obs,done[], update , is_reconstruction
                states, loss_dict, reconstruction_dict = model.train_world(path_obs, path_act, path_rew, preferred_obs,
                                                                           path_done=path_done,
                                                                           update_target=update_target,
                                                                           get_reconstruction=log_images)
                # 传入（时间步，策略网络，后验状态st，pre obs,obs_path）
                policy_loss_dict = model.train_policy_value(horizon, model.policy, states, preferred_obs,
                                                            obs_batch=path_obs)  # 策略、价值网络训练

                for k in loss_dict:
                    if k in main_loss_dict:
                        main_loss_dict[k] += loss_dict[k] / n_epochs
                    else:
                        main_loss_dict[k] = loss_dict[k] / n_epochs

                for k in policy_loss_dict:
                    if k in main_policy_loss_dict:
                        main_policy_loss_dict[k] += policy_loss_dict[k] / n_epochs
                    else:
                        main_policy_loss_dict[k] = policy_loss_dict[k] / n_epochs
            # Logging
            for k, v in main_loss_dict.items():
                writer.add_scalar('world_model/' + k, v, global_step=current_step)
            if len(reconstruction_dict) > 0:
                print('Logging videos')
            for k, v in reconstruction_dict.items():  # 不需要对图像进行重构，没有重构字典
                writer.add_video(k, v.permute(1, 0, 2, 3, 4), global_step=current_step, fps=15)
            for k, v in main_policy_loss_dict.items():
                writer.add_scalar('policy/' + k, v, global_step=current_step)
            if model.reinforce or not config['use_rewards']:
                print('Updating Value Model Target...')
                model.update_target_network(1., model.value_model, model.value_target)  # 软更新网络
            # Save model
            if config['save_every'] > 0 and tot_episodes % config['save_every'] == 0:
                save_model(model, str(logdir / f'world_model.pt'))

        done = False
        # 世界模型和动作策略模型 训练完毕，开始进行策略选取，选取直到episode做完
        while not done:
            # Policy Selection
            if current_step // 100 > century:  # 每一百轮打印一遍
                print('Step: ', current_step)
                century += 1
            # 写入  策略预测长度，策略模型，偏好观察，前一个状态（首次前一个状态为none）   返回 策略概率分布，策略选择动作，期望自由能字典，当前为固定动作即返回确定动作
            policy_distr, policy_actions, future_loss_dict = model.policy_distribution(policy_lookahead, [model.policy],
                                                                                       preferred_obs, prev_state)

            for k in future_loss_dict:
                if k in policy_dict:
                    policy_dict[k] += future_loss_dict[k]
                else:
                    policy_dict[k] = future_loss_dict[k]  # 写入字典

            # Action 执行选取出来的动作
            for policy_step in range(policy_lookahead):  # 对该策略进行操作，执行过程为 lookahead步
                action_index = policy_distr.sample()  # 对自由能最高的动作字典进行采样，采样出下标
                action = policy_actions[action_index][policy_step].cpu()  # 选出对应下标的动作，即采样出来的动作
                act_noise = torch.randn(*action.shape, device=action.device) * expl_amount  # expl_amount噪声数值？
                action = torch.clamp(action + act_noise, -1, 1)
                # 执行策略网络选取出来的动作
                timestep, rew, done, info = env.step(action.numpy())
                if config['render_every'] > 0 and tot_episodes % config['render_every'] == 0:
                    env.render()  # 返回图像矩阵
                # 观察值标准化
                obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])
                # 记录reward
                rew_tensor = torch.Tensor([rew]).to(device)
                # Minigrid only: to simplify reward prediction, as the agent has no knowledge of time 简化reward预测，agent没有时间方面的知识
                done_tensor = torch.Tensor([done]).to(device)  # 是否完成epi
                current_obs = obs.expand(1, 1, *obs.shape)  # 记录昨晚动作后的当前状态
                # 求当前状态的 通过后验模型 后验st
                prev_state = model.step(current_obs.to(device), torch.Tensor([rew]).reshape(1, 1, 1).to(device),
                                        action.view(1, 1, *action.shape).to(device), prev_state)
                # 计算当前状态的 自由能  使用st
                free_energy = model.eval_obs(obs, rew_tensor, preferred_obs, prev_state)
                episode['obs'].append(obs.cpu())
                episode['act'].append(action.cpu())
                episode['rew'].append(rew_tensor.cpu())
                episode['done'].append(done_tensor.cpu())
                episode['free_energy'].append(free_energy.cpu())

                current_step += 1 * action_repeat  # 执行步数+1
                cur_return += rew  # 当前总奖励

                if done:  # 如果该epi完成，总epi+1
                    tot_episodes += 1
                    # 记录全局episode数据
                    episode_store.add(np.stack(episode['obs'], axis=0), np.stack(episode['act'], axis=0),
                                      np.stack(episode['rew'], axis=0), np.stack(episode['free_energy'], axis=0),
                                      np.stack(episode['done'], axis=0))
                    # print('Step: ', current_step, ' Return: ', np.round(cur_return, 2), 'Expected Free Energy: ', np.round(policy_dict['policy_expected_loss'],2) )
                    writer.add_scalar('environment/return', cur_return, global_step=current_step)
                    writer.add_scalar('environment/return_over_episodes', cur_return, global_step=tot_episodes)
                    writer.add_scalar('environment/free_energy', np.sum(episode['free_energy']),
                                      global_step=current_step)
                    for k in policy_dict:
                        writer.add_scalar('environment/' + k, policy_dict[k], global_step=current_step)

                    # Record and/or eval
                    if config['record_every'] > 0 and tot_episodes % config['record_every'] == 0:
                        writer.add_video('environment/train_episode',
                                         np.expand_dims(np.stack(episode['obs'], axis=0) + 0.5, 0),
                                         global_step=tot_episodes, fps=15)
                    # 清空局部episode
                    episode = dict(obs=[], act=[], rew=[], free_energy=[], done=[])
                    timestep = env.reset()  # 重置环境
                    if config['render_every'] > 0 and tot_episodes % config['render_every'] == 0:
                        env.render()  # 渲染状态和环境
                    # 初始化obs标准化
                    obs = wrappers.get_scaled_obs(timestep, device=device, is_minigrid='minigrid' in config['suite'])
                    # 记录
                    episode['obs'].append(obs.cpu())
                    # 重置局部内容
                    prev_state = None
                    cur_return = 0
                    policy_dict = dict()
                    # This breaks the policy_step cycle
                    break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--disable-cuda', help='disable cuda', action='store_true', default=False)
    parser.add_argument('--save-every', help='save model', default=100, type=int)
    parser.add_argument('--render-every', help='render agent', default=0, type=int)
    parser.add_argument('--record-every', help='record training episodes', default=100, type=int)
    parser.add_argument('--recon-every', help='log reconstructions as videos', default=100, type=int)
    # easy reacher
    parser.add_argument('--suite', help='suite name', default='dmc')
    parser.add_argument('--task', help='task name', default='reacher_easy_13')
    parser.add_argument('--algo', help='algo name', default='contrastive_actinf')
    parser.add_argument('--config', nargs='+', help='configuration name', default=['dmc_small', 'dmc_benchmark'])
    parser.add_argument('--seed', help='set random seed', default=34, type=int)

    args = parser.parse_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / 'configs.yaml').read_text())  # 读进yaml中所有参数
    conf = dict()
    for k in ['base', *args.config]:
        conf = dict(conf, **configs[k])
    conf = dict(conf, **configs['algos'][args.algo])
    for k in args.__dict__:
        conf[k] = args.__dict__[k]
    print(conf)
    main(conf)
