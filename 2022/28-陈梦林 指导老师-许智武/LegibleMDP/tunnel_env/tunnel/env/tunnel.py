import time

import gym
from gym.spaces import Discrete, Box
import numpy as np
import matplotlib.pyplot as plt, matplotlib
from matplotlib import cm
import matplotlib.patches as mpatches


class TunnelBuilder:

    def __init__(self):
        self.tunnel = None
        self.clear_on_add = None

    @property
    def size(self):
        """

        :return: width, length, colors
        """
        return self.tunnel.shape

    def new_tunnel(self, width, length, colors, clear_on_add=False):
        """
        :param width: width of the tunnel
        :param length: length of the tunnel
        :param colors: number of colors that cells can have
        :param clear_on_add: whether to clear all the colors for the cells of a newly added shape
        :return:
        """
        self.clear_on_add = clear_on_add
        self.tunnel = np.zeros((width, length, colors + 1))
        return self

    def add_shape(self, x, y, color, shape_fcn, clear=None):
        cells = shape_fcn(self.tunnel, x, y)
        clear = self.clear_on_add if clear is None else clear
        if clear:
            cells[:, :, :] = 0
        cells[:, :, color] = 1
        return self

    def add_obstacle(self, x, y, shape_fcn):
        """
        `add_obstacle` adds an obstacle to the grid

        :param x: x coordinate of the center of the shape
        :param y: the y coordinate of the top left corner of the shape
        :param shape_fcn: a function that takes a single argument, the size of the grid, and returns a list of tuples, each
        of which is a coordinate in the grid
        :return: The return value is the number of cells that were cleared.
        """
        """
        `add_obstacle`向网格添加障碍
        ：param x：形状中心的x坐标
        ：param y：形状左上角的y坐标
        ：param shape_fcn：一个函数，它接受一个参数，即网格的大小，并返回元组列表，每个元组
        其中是网格中的坐标
        ：return：返回值是已清除的单元格数。
        """
        w, l, c = self.size
        return self.add_shape(x, y, c - 1, shape_fcn, clear=True)

    def add_obstacle_random(self, shape_fcn):
        w, l, c = self.size
        x = np.random.randint(w)
        y = np.random.randint(l)
        self.add_obstacle(x, y, shape_fcn)
        return self

    def add_shape_random(self, shape_fcn):
        w, l, c = self.size
        x = np.random.randint(w)
        y = np.random.randint(l)
        c = np.random.randint(c - 1)
        self.add_shape(x, y, c, shape_fcn)
        return self

    @staticmethod
    def rect(grid, x, y, sx, sy):
        return grid[x:x + sx, y:y + sy, :]

    @staticmethod
    def line_h(grid, x, y, length):
        return TunnelBuilder.rect(grid, x, y, 1, length)

    @staticmethod
    def line_w(grid, x, y, length):
        return TunnelBuilder.rect(grid, x, y, length, 1)

    @staticmethod
    def square_2(grid, x, y):
        return TunnelBuilder.rect(grid, x, y, 2, 2)

    @staticmethod
    def square_3(grid, x, y):
        return TunnelBuilder.rect(grid, x, y, 3, 3)

    @staticmethod
    def square_4(grid, x, y):
        return TunnelBuilder.rect(grid, x, y, 4, 4)

    @staticmethod
    def square_5(grid, x, y):
        return TunnelBuilder.rect(grid, x, y, 5, 5)

    @staticmethod
    def line_h_5(grid, x, y):
        return TunnelBuilder.line_h(grid, x, y, 5)

    @staticmethod
    def line_h_2_5(grid, x, y):
        return TunnelBuilder.rect(grid, x, y, 2, 5)

    @staticmethod
    def line_h_10(grid, x, y):
        return TunnelBuilder.line_h(grid, x, y, 10)

    @staticmethod
    def line_h_2_10(grid, x, y):
        return TunnelBuilder.rect(grid, x, y, 2, 10)

    @staticmethod
    def line_w_5(grid, x, y):
        return TunnelBuilder.line_w(grid, x, y, 5)

    @staticmethod
    def line_w_10(grid, x, y):
        return TunnelBuilder.line_w(grid, x, y, 10)

    @staticmethod
    def shapes():
        return TunnelBuilder.square_2, TunnelBuilder.square_3, TunnelBuilder.square_4, TunnelBuilder.square_5, \
               TunnelBuilder.line_h_5, TunnelBuilder.line_h_10, TunnelBuilder.line_h_2_5, TunnelBuilder.line_h_2_10, \
               TunnelBuilder.line_w_5, TunnelBuilder.line_w_10


class TunnelEnv(gym.Env):

    def __init__(self, tunnel_length, tunnel_width, sight_distance,
                 num_colors, reward_color, color_reward_amount, obstacle_reward, steer_reward,
                 reset_on_obstacle_hit
                 ):
        """

        :param tunnel_length: length of the tunnel (L)
        :param tunnel_width: height of the tunnel (H)
        :param num_colors: number of colors of which cells can be colored of
        :param reward_color: number of the rewarding color. The agent receives a reward of 'reward_amount' every time it
        is on a cell of this color. If None the rewarding color is randomly selected at every episode
        :param sight_distance: sight distance of the agent
        The functions should a part of the tunnel depending also on the provided coordinates x and y. See TunnelBuilder for examples
        :param color_reward_amount: the reward the agent receives by passing on a cell of a rewarding color
        :param obstacle_reward: reward of entering a obstacle cells.
        :param steer_reward: reward obtained for using left or right action (should be 0 or small negative)
        :param reset_on_obstacle_hit: whether the episodes are reset when the agent hits an obstacle

        ：param tunnel_length：隧道长度（L）
        ：param tunnel_width：隧道的高度（H）
        ：param num_colors: 单元格可以着色的颜色数
        ：param reward_color：奖励颜色的编号。代理每次都会收到“reward_amount”的奖励
        在这种颜色的单元格上。如果没有，则在每集随机选择奖励颜色
        ：param sight_distance：代理的视距
        函数应该是隧道的一部分，这也取决于提供的坐标x和y。有关示例，请参见TunnelBuilder
        ：param color_reward_amount：代理通过传递奖励颜色的单元格而获得的奖励
        ：param obstruacle_reward: 进入障碍单元格的奖励。
        ：param steel_reward：使用左或右动作获得的奖励（应为0或小负值）
        ：param reset_on_obstacle_hhit：当代理遇到障碍物时，是否重置剧集
        """
        self.tunnel_length = tunnel_length
        self.tunnel_width = tunnel_width
        self.sight_distance = sight_distance

        self.n_colors = num_colors

        self.base_reward_color = reward_color
        self.episode_reward_color = reward_color
        self.reward_amount = color_reward_amount

        self.obstacle_reward = obstacle_reward
        self.steer_reward = steer_reward
        self.accumulated_reward = 0

        self.reset_on_obstacle_hit = reset_on_obstacle_hit

        self.pos = None
        self.tunnel = None

        self.plot = None
        self.image = None

        self.action_space = Discrete(3)
        self.observation_space = Box(-np.inf, np.inf, shape=self.state_shape)

        self.reset_tunnel = True

        self._render_mask = None
        self._accum_render_mask = None

    @property
    def state_shape(self):
        return self.tunnel_width, self.sight_distance, self.n_colors * 2 + 1,

    @property
    def size(self):
        return self.tunnel.shape[:-1]

    @property
    def reward_color(self):
        return self.episode_reward_color

    def reset(self):
        if self.reset_tunnel or self.tunnel is None:
            self.tunnel = self.new_tunnel()

        self.pos = np.array(self.get_initial_position()).reshape(-1)
        assert len(self.pos) == 2

        if self.base_reward_color is None:
            self.episode_reward_color = np.random.randint(self.n_colors)
        else:
            self.episode_reward_color = self.base_reward_color

        self.accumulated_reward = 0

        self._render_mask = np.ones((self.tunnel_width, self.tunnel_length,))

        if self._accum_render_mask is None or self.reset_tunnel:
            self._accum_render_mask = np.zeros((self.tunnel_width, self.tunnel_length,))

        s = self.get_state()
        return s

    def new_tunnel(self):
        """
        :return: A tunnel of shape W x L x (2*C+1)
        """
        raise NotImplementedError

    def step(self, action):
        self.pos = self.get_new_pos(action)

        state = self.get_state()

        reward = self.get_step_reward(action)
        done = self.get_step_done()
        info = self.get_info()
        self.accumulated_reward += self.get_reward(self.episode_reward_color)
        return state, reward, done, info

    def get_step_done(self):
        x, y = self.pos
        return ((self.pos[1] + self.sight_distance) >= self.tunnel_length) or \
               (self.check_obstacle(x, y) and self.reset_on_obstacle_hit)

    def get_step_reward(self, action):
        x, y = self.pos
        reward = 0

        # Obstacle reward
        reward += float(self.check_obstacle(x, y)) * self.obstacle_reward

        # Cell reward
        reward += self.get_reward(self.episode_reward_color)

        # Action reward
        reward += float(action != 1) * self.steer_reward
        return reward

    def get_initial_position(self):
        return np.random.choice(self.tunnel_width), 0,

    def get_new_pos(self, action):
        new_pos = np.array(self.pos)
        if action == 0:  # UP
            new_pos[0] = max(0, new_pos[0] - 1)
        elif action == 1:  # CENTER
            new_pos[0] = new_pos[0]
        elif action == 2:  # DOWN
            new_pos[0] = min(self.size[0] - 1, new_pos[0] + 1)
        else:
            raise ValueError(f'Unsupported action {action}')
        new_pos[1] += 1
        return new_pos

    def get_state(self):
        """
        H : height, S: sight distance C: num colors
        :return: {
        colors : H x S x C,
        obstacles : H x S
        reward_color : int
        position : H x S
        }
        """
        """
        H：高度，S：视距 C:num颜色
        ：返回：{
        颜色：H x S x C，
        障碍物：H x S
        reward_color:int奖励颜色的编号。
        位置：H x S
        }
        """
        px, py = self.pos
        # Taking the tunnel and slicing it to get the colors.
        # 把隧道切开，得到颜色。
        colors = self.tunnel[:, py:py + self.sight_distance, :self.n_colors]
        # Taking the tunnel and slicing it to get the obstacles.
        obstacles = self.tunnel[:, py:py + self.sight_distance, self.n_colors]
        position = np.zeros(colors.shape[:-1])
        position[px, :] = 1
        return {
            'colors':colors,
            'obstacles': obstacles,
            'reward_color': str(self.episode_reward_color),
            'position':position,
        }


    def get_max_reward(self, color):
        r = self.tunnel[:, :-self.sight_distance, color].max(axis=0)
        r = r.sum()
        return r

    def check_obstacle(self, x, y):
        cell = self.tunnel[x, y, :]
        return cell[-1] == 1

    def get_reward(self, color):
        x, y = self.pos
        cell = self.tunnel[x, y, :]
        hit = float(cell[color] == 1)

        cell_reward = hit * self.reward_amount

        return cell_reward

    def get_info(self):
        info = {
            'accumulated_reward': self.accumulated_reward,
            'max_accumulated_reward': self.get_max_reward(self.episode_reward_color),
        }
        return info

    # RENDERING FUNCTIONS

    def render(self, mode='human'):
        if self.plot is None:
            fg = plt.figure()
            ax = fg.gca()
            data = np.zeros(self.size)

            cmap = cm.get_cmap('tab10')
            colors = list(cmap.colors)
            colors[0] = (0, 0, 0)
            # colors[1] = (0, 0, 0)
            # colors[2] = (0, 0, 0)
            cmap.colors = colors
            self.plot = ax.imshow(data, vmin=0, vmax=self.n_colors + 1, cmap=cmap)

            patches = self.render_legend_patches(cmap)
            self.plot.axes.legend(handles=patches, bbox_to_anchor=(-0.15, 0.95), loc=2, borderaxespad=0.)
            self.plot.axes.set_title('Tunnel Environment')
            self.plot.axes.get_xaxis().set_ticks([])
            self.plot.axes.get_yaxis().set_ticks([])

        data = self.get_imshow_data()
        self.plot.set_data(data)

        render_mask = np.copy(self._render_mask)
        x, y = self.pos
        render_mask[x, y] = 1
        self.plot.set_alpha(render_mask)

        plt.draw()
        plt.pause(1e-3)

    def get_imshow_data(self, show_position=True):
        tunnel = self.tunnel  # self.tunnel[:,:-self.sight_distance+1,:]
        data = np.argmax(tunnel, -1)
        no_color = (tunnel == 0).all(axis=-1)
        data[~no_color] += 1

        if show_position:
            px, py = self.pos
            data[px, py] = self.episode_reward_color + 1
        data = data[:, :-self.sight_distance + 1]
        return data

    def set_render_mask(self, mask):
        # mask = np.array(mask > 0, dtype='bool')
        assert len(mask.shape) == 2
        mx, my = mask.shape
        tx, ty = self.size
        assert mx == tx and my == ty
        self._render_mask = mask
        self._accum_render_mask += mask

    def set_render_mask_sight_window(self, mask):
        m = np.zeros(self.size)
        x, y = self.pos
        m[:, y:y + self.sight_distance] = mask
        return self.set_render_mask(m)

    def render_legend_patches(self, cmap):
        data = np.argmax(self.tunnel, -1)
        no_color = (self.tunnel == 0).all(axis=-1)
        data[~no_color] += 1
        px, py = self.pos
        data[px, py] = self.episode_reward_color + 1
        values = np.unique(data).astype('int')
        colors = [cmap(self.plot.norm(value)) for value in values]
        types = {0: 'empty', 1: 'color-0', 2: 'color-1', 3: 'color-2', 4: 'color-3', 5: 'obstacle'}
        patches = [mpatches.Patch(color=colors[i], label=f'{types[values[i]]}') for i in range(len(values))]
        return patches

    def save_plot(self, filename):
        matplotlib.use("pgf")
        matplotlib.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

        self.plot.figure.savefig(f'{filename}.pgf')


class BuilderTunnelEnv(TunnelEnv):

    def __init__(self, tunnel_length, tunnel_width, sight_distance,
                 num_colors, reward_color, color_density, color_shapes, color_reward_amount,
                 obstacle_density, obstacle_shapes, obstacle_reward, steer_reward=0, reset_on_obstacle_hit=False
                 ):
        """
        :param color_density: number of colored shapes among all colors. Colors are randomly selected
        :param color_shapes: list of fcn(grid, x, y) functions. grid is the H x L numpy array representing the tunnel.
        The functions should a part of the tunnel depending also on the provided coordinates x and y. See TunnelBuilder for examples
        :param obstacle_density: number of obstacles
        :param obstacle_shapes: shapes of obstacles. same as 'color_shapes'

        """
        super().__init__(tunnel_length, tunnel_width, sight_distance,
                         num_colors, reward_color, color_reward_amount, obstacle_reward, steer_reward,
                         reset_on_obstacle_hit
                         )

        self.color_density = color_density
        self.color_shapes = color_shapes

        self.obstacle_density = obstacle_density
        self.obstacle_shapes = obstacle_shapes
        self.accumulated_reward = 0

        self.clear_on_add = False

    def new_tunnel(self):
        b = TunnelBuilder().new_tunnel(self.tunnel_width, self.tunnel_length, self.n_colors,
                                       clear_on_add=self.clear_on_add)
        for i in range(self.color_density):
            s = np.random.randint(len(self.color_shapes))
            b.add_shape_random(self.color_shapes[s])

        for i in range(self.obstacle_density):
            s = np.random.randint(len(self.obstacle_shapes))
            b.add_obstacle_random(self.obstacle_shapes[s])

        # return np.load('./tunnel.npy')
        return b.tunnel

    def get_initial_position(self):
        return 6, 0


class TwoGoalsTunnelEnv(TunnelEnv):

    def __init__(self, tunnel_length, tunnel_width, sight_distance, reset_on_obstacle_hit):

        super().__init__(tunnel_length, tunnel_width, sight_distance,
                         num_colors=2, reward_color=0, color_reward_amount=1, obstacle_reward=-5, steer_reward=0,
                         reset_on_obstacle_hit=reset_on_obstacle_hit
                         )

        self.pos1 = (4, 20)
        self.pos2 = (4, 10)
        self.obstacles = None
        self.goal_size = 2

    def new_tunnel(self):
        tunnel = np.zeros(shape=(self.tunnel_width, self.tunnel_length, self.n_colors + 1))

        self.place_goal(tunnel, self.pos1, self.episode_reward_color)
        self.place_goal(tunnel, self.pos2, self.episode_reward_color + 1)
        if self.obstacles is not None:
            for p in self.obstacles:
                self.place_goal(tunnel, p, self.n_colors)

        return tunnel

    def get_initial_position(self):
        return 6, 0

    def place_goal(self, tunnel, pos, color):
        if pos is None:
            return
        x1, y1 = pos
        c = TunnelBuilder.rect(tunnel, x1, y1, self.goal_size, self.goal_size)
        c[:, :, color] = 1

