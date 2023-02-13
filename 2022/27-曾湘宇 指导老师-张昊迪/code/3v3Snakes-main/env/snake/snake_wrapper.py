
from chooseenv import make
from copy import deepcopy
from common import *



class SnakeEnvWrapper():
    def __init__(self, mode='OneVsOne-v0'):
        self.env = make("snakes_3v3", conf=None)
        self.states = None
        self.ctrl_agent_index = [0, 1, 2]
        self.obs_dim = 26
        self.height = self.env.board_height
        self.width = self.env.board_width
        self.episode_reward = np.zeros(6)

    def act(self):
        return self.env.act(self.states)

    def reset(self):
        states = self.env.reset()
        length = []
        #基本观测结果 包括队友和对手
        obs = process_obs_joint(states[0])
        #emmm一种copy方法
        self.states = deepcopy(states)
                        
        legal_action = get_legal_actions(states[0])
        info = {}
        #信息set中legal_action部分完成
        info ["legal_action"] = legal_action
        # 返回的是观测矩阵和info（包括合法动作）
        return obs, info

    def step(self, actions):
        next_state, reward, done, _, info = self.env.step(self.env.encode(actions))
        # 根据新的情况初始化地图？
        next_obs = process_obs_joint(next_state[0])
        # reward shaping
        step_reward = get_reward_joint(reward, next_state[0], done)
        length = []
        # 更新蛇长？
        for i in range(2,8):
            length.append(len(next_state[0][i]))
        info ["length"] = length
        # 根据新情况计算合法行为
        legal_action = get_legal_actions(next_state[0])
        info ["legal_action"] = legal_action
        # 返回更新后的结果
        return next_obs, step_reward, done, info

    def render(self):
        self.env.render()


