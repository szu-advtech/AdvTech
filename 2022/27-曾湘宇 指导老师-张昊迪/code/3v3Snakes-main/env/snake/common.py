import numpy as np
import torch
import torch.nn as nn
import math
import copy
from typing import Union
from torch.distributions import Categorical
import os
import yaml

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")


def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(
            tgt_param.data * (1.0 - tau) + src_param.data * tau
        )


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}


def mlp(sizes,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity'):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    return nn.Sequential(*layers)

def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map

def get_min_bean(x, y, beans_position):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index

def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position)
        beans_position.pop(index)

        next_distances = []
        up_distance = math.inf if head_surrounding[0] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append(up_distance)
        down_distance = math.inf if head_surrounding[1] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append(down_distance)
        left_distance = math.inf if head_surrounding[2] > 1 else \
            math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(left_distance)
        right_distance = math.inf if head_surrounding[3] > 1 else \
            math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(right_distance)
        actions.append(next_distances.index(min(next_distances)))
    return actions


# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
def get_observations(state, agents_index, obs_dim, height, width):
    state_copy = state[0].copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    observations = np.zeros((3, obs_dim))
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    for i in agents_index:
        # self head position
        observations[i][:2] = snakes_position[i][0][:]

        # head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, i, 0)
        observations[i][16:] = snake_heads.flatten()[:]
    return observations


def get_reward(info, snake_index, reward, score):
    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))
    for i in snake_index:
        if score == 1:
            step_reward[i] += 50
        elif score == 2:
            step_reward[i] -= 25
        elif score == 3:
            step_reward[i] += 10
        elif score == 4:
            step_reward[i] -= 5

        if reward[i] > 0:
            step_reward[i] += 20
        else:
            self_head = np.array(snake_heads[i])
            dists = [np.sqrt(np.sum(np.square(other_head - self_head))) for other_head in beans_position]
            step_reward[i] -= min(dists)
            if reward[i] < 0:
                step_reward[i] -= 10

    return step_reward


def logits_random(act_dim, logits):
    logits = torch.Tensor(logits).to(device)
    acs = [Categorical(out).sample().item() for out in logits]
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions


def logits_greedy(state, logits, height, width):
    state_copy = state[0].copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state = np.array(snake_map)
    state = np.squeeze(snake_map, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

    logits = torch.Tensor(logits).to(device)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])

    greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])

    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = greedy_action

    return action_list


def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()

def process_observations_each(state, ctrl_agent_index):
    """输入官方obs，官方环境的state[0],输出网络输入（单个agent（6,10,20,5））"""
    board_width = state['board_width']
    board_height = state['board_height']
    base_map = np.array([[[0] for _ in range(board_width)]
                  for _ in range(board_height)]) #这里的处理方式 初始化地图

    beans_map = base_map.copy() # 赋值方法 复制到beans地图中
    for bean_pos in state[1]: # 赋值给豆子
        beans_map[bean_pos[0]][bean_pos[1]][0] = 1
    # copy 多几份
    snakes_map = base_map.copy()
    team_head_map = base_map.copy()
    enemy_head_map = base_map.copy()
    my_head_map = base_map.copy()
    my_snake_map = base_map.copy()
    teammate_map = []
    enemy_map = []
    # 有两个队友 就copy两份
    for i in range(2):
        teammate_map.append(base_map.copy())
    # 有三个对手 就copy三份
    for i in range(3):
        enemy_map.append(base_map.copy())
    # 转换成张量形式
    teammate_map = np.array(teammate_map)
    enemy_map = np.array(enemy_map)
    team_idx = None
    enemy_idx = None
    if ctrl_agent_index in [0,1,2]:
        team_idx = [0,1,2]
        enemy_idx = [3,4,5]
    else:
        team_idx = [3,4,5]
        enemy_idx = [0,1,2]
    # 相减即 取一个集合中另一集合没有的元素
    teammate_idx = list(set(team_idx)-set([ctrl_agent_index]))
    teammate_idx = sorted(teammate_idx)
    #给相应地图赋值
    for i in range(2,8):
        if i-2 == ctrl_agent_index:
            my_head_map[state[i][0][0]][state[i][0][1]][0] = 1
        if i-2 in team_idx:
            team_head_map[state[i][0][0]][state[i][0][1]][0] = 1
        else:
            enemy_head_map[state[i][0][0]][state[i][0][1]][0] = 1
        for snake_pos in state[i]:
            if i-2 == ctrl_agent_index:
                my_snake_map[snake_pos[0]][snake_pos[1]][0] = 1
            if i-2 == teammate_idx[0]:
                teammate_map[0][snake_pos[0]][snake_pos[1]][0] = 1
            if i-2 == teammate_idx[1]:
                teammate_map[1][snake_pos[0]][snake_pos[1]][0] = 1
            if i-2 == enemy_idx[0]:
                enemy_map[0][snake_pos[0]][snake_pos[1]][0] = 1
            if i-2 == enemy_idx[1]:
                enemy_map[1][snake_pos[0]][snake_pos[1]][0] = 1
            if i-2 == enemy_idx[2]:
                enemy_map[2][snake_pos[0]][snake_pos[1]][0] = 1
            snakes_map[snake_pos[0]][snake_pos[1]][0] = 1
    #返回拼接的结果
    return np.concatenate((snakes_map, my_head_map, team_head_map, enemy_head_map, beans_map, my_snake_map, teammate_map[0], teammate_map[1], enemy_map[0], enemy_map[1], enemy_map[2]), axis=-1)

def process_obs_joint(state):
    board_width = state['board_width']
    board_height = state['board_height']
    joint_obs = np.zeros((6, board_height, board_width, 11))
    #对每条蛇都初始化地图
    for i in range(6):
        joint_obs[i] = process_observations_each(state, i)
    return joint_obs

# def my_get_reward(flag, reward, index, emy_idx, tau):
#     """输入官方reward(6,)，蛇编号，敌方阵营（default：[3,4,5], team_spirit参数），输出reward（scalar）"""
#     assert index not in emy_idx, "Index Error!\n"
#     emy_reward = sum([reward[i] for i in emy_idx])/3
#     # print("emy_reward:{}".format(emy_reward))
#     total_idx = set([0,1,2,3,4,5])
#     team_idx = list(total_idx-set(emy_idx))
#     # print("team_idx:{}".format(team_idx))
#     team_reward = sum([reward[i] for i in team_idx])/3
#     # print("team_reward:{}".format(team_reward))
#     true_reward = (reward[index] - emy_reward) * (1-tau) + team_reward * tau
#     # print("get_reward, true_reward:{}".format(true_reward))
#     if flag==1:
#         true_reward += 10
#     if flag==-1:
#         true_reward -= 10
#     return true_reward

# def get_reward_joint(reward, state, done):
#     len1 = 0
#     len2 = 0
#     for i in range(2, 5):
#         len1 += len(state[i])
#     for i in range(5, 8):
#         len2 += len(state[i])
#     win1 = len1>len2
#     win2 = len2>len1
#     flag1 = 0
#     flag2 = 0
#     if done and win1:
#         flag1 = 1
#         flag2 = -1
#     if done and win2:
#         flag2 = 1
#         flag1 = -1
#     true_reward = np.zeros((6,))
#     for i in range(3):
#         true_reward[i] = my_get_reward(flag1, reward, i, [3,4,5], 0.3)
#     for i in range(3,6):
#         true_reward[i] = my_get_reward(flag2, reward, i, [0,1,2], 0.3)
#     return true_reward
    
def get_reward_joint(reward, state, done):
    len1 = 0
    len2 = 0
    for i in range(2, 8):
        if i in [2,3,4]:
            len1 += len(state[i])
        else:
            len2 += len(state[i])
    flag = 0 # 0代表平，-1代表0，1，2胜，1代表3，4，5胜
    if done:
        if len1>len2:
            flag = -1
        elif len1<len2:
            flag = 1
    true_reward = np.zeros((6,))
    true_reward[0] = reward[0] - (reward[3] + reward[4] + reward[5]) / 3.0
    true_reward[1] = reward[1] - (reward[3] + reward[4] + reward[5]) / 3.0
    true_reward[2] = reward[2] - (reward[3] + reward[4] + reward[5]) / 3.0
    true_reward[3] = reward[3] - (reward[0] + reward[1] + reward[2]) / 3.0
    true_reward[4] = reward[4] - (reward[0] + reward[1] + reward[2]) / 3.0
    true_reward[5] = reward[5] - (reward[0] + reward[1] + reward[2]) / 3.0

    tau = 0.8
    true_reward_2 = np.zeros((6,))
    true_reward_2[0] = true_reward[0] * tau + (true_reward[0] + true_reward[1] + true_reward[2]) / 3.0 * (1 - tau)
    true_reward_2[1] = true_reward[1] * tau + (true_reward[0] + true_reward[1] + true_reward[2]) / 3.0 * (1 - tau)
    true_reward_2[2] = true_reward[2] * tau + (true_reward[0] + true_reward[1] + true_reward[2]) / 3.0 * (1 - tau)
    true_reward_2[3] = true_reward[3] * tau + (true_reward[3] + true_reward[4] + true_reward[5]) / 3.0 * (1 - tau)
    true_reward_2[4] = true_reward[4] * tau + (true_reward[3] + true_reward[4] + true_reward[5]) / 3.0 * (1 - tau)
    true_reward_2[5] = true_reward[5] * tau + (true_reward[3] + true_reward[4] + true_reward[5]) / 3.0 * (1 - tau)

    final_reward = 10
    if flag == -1:
        true_reward_2[0] += final_reward
        true_reward_2[1] += final_reward
        true_reward_2[2] += final_reward
        true_reward_2[3] -= final_reward
        true_reward_2[4] -= final_reward
        true_reward_2[5] -= final_reward
    elif flag == 1:
        true_reward_2[0] -= final_reward
        true_reward_2[1] -= final_reward
        true_reward_2[2] -= final_reward
        true_reward_2[3] += final_reward
        true_reward_2[4] += final_reward
        true_reward_2[5] += final_reward
    return true_reward_2


def choose_action(self, obs, last_direction):
    """输入obs（6,5,10,20），官方state[0]["last_direction"]，输出action，已屏蔽非法action"""
    opposite_action_list = {"up":1, "down":0, "left":3, "right":2}
    if(last_direction):
        opposite_direction = [opposite_action_list[last_direction[i]] for i in range(6)]
    assert obs.shape[0] == self.num_agent*2, "Not correct shape for obs, shape[0] should be num_agent"
    """
    input: shape* obs
    output: shape* (actions, action_log_prob, value)
    actions: scalar
    action_log_probs: scalar
    value: scalar
    """
    # avoid illegal action
    # print("choose actions, obs.shape:", obs.shape)
    p = np.random.random()
    obs = torch.Tensor(obs).to(self.device)
    logits, value = self.network(obs)
    if(last_direction):
        for i in range(6):
            logits[i][opposite_direction[i]] = float('-inf')
    # print("choose actions, logits.shape:{}, value shape:{}".format(logits.shape, value.shape))
    if p > self.eps:

        actions = [Categorical(logits=logits[i]).sample()
                    for i in range(len(logits))]
    else:
        actions = [np.random.randint(self.act_dim)
                    for i in range(len(logits))]
    actions = torch.tensor(actions)
    action_log_probs = [Categorical(
        logits=logits[i]).log_prob(actions[i]) for i in range(len(actions))]

    action_log_probs = torch.tensor(action_log_probs)

    # print("choose actions, actions.shape:{}, action_log_probs.shape:{}".format(actions.shape, action_log_probs.shape))

    self.eps *= self.decay_speed

    return actions.view(-1, 1), action_log_probs.view(-1, 1), value.view(-1, 1)
    
def get_legal_actions(state):
    '''
    legal_actions = [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]

    '''
    legal_actions = [[1 for i in range(4)] for j in range(6)]
    head_pos = []
    board_width = state['board_width']
    board_height = state['board_height']
    base_map = ([[0 for _ in range(board_width)]
                  for _ in range(board_height)])

    snakes_map = base_map.copy()
    for i in range(2,8):
        for snake_pos in state[i][:-1]:
            snakes_map[snake_pos[0]][snake_pos[1]] = 1
        head_pos.append(state[i][0])
    #标注方向
    dir = {0:[-1,0],1:[1,0],2:[0,-1],3:[0,1]}
    #看各个位置哪个位置是不合格的 把1->0
    for i in range(6):
        for j in range(4):
            pos = [0,0]
            pos[0] = (head_pos[i][0] +  dir[j][0]) % board_height
            pos[1] = (head_pos[i][1] + dir[j][1]) % board_width
            if snakes_map[pos[0]][pos[1]]:
                legal_actions[i][j] = 0
    # 返回遍历之后合格 的矩阵
    return legal_actions
