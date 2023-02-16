# 培训代理
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import itertools
from mdptoolbox_git import mdp


def get_one_hot(size, pos, reshape=None):
    """
    It creates a one-hot vector of size `size` with a 1. at position `pos`

    :param size: the size of the one-hot vector
    :param pos: the position of the one-hot vector that you want to be 1
    :param reshape: if not None, reshape the one-hot vector to this shape
    :return: A one-hot vector of size 'size' with a 1 at position 'pos'
    """
    """
    它创建一个大小为“size”的单热向量，其位置为“pos”`
    ：param size：一个热向量的大小
    ：param pos：要成为1的一个热向量的位置
    ：param重塑：如果不是“无”，则将一个热向量重塑为此形状
    ：return：大小为“size”且位置为“pos”的1热向量
    """
    a = np.zeros(size)
    a[pos] = 1.
    if reshape is not None:
        a = a.reshape(reshape)
    return a


def find_x_y(s):
    """
    It finds the first row and column that contains a 1

    :param s: the state of the game
    :return: The x and y coordinates of the first 1 in the array.
    """
    """
    它查找包含1的第一行和列
    ：param s：游戏状态
    ：return：数组中第一个1的x和y坐标。
    """
    x = np.argwhere(np.sum(s==1, 1)).flatten()[0]
    y = np.argwhere(np.sum(s==1, 0)).flatten()[0]
    return x, y


def get_P_A(X, dx, dy):
    """
    :param X: (S, S) matrix with a single cell equal 1
    :param dx: movement along x
    :param dy: movement along y
    :return: the transition matrix of size (S,S)
    """
    """
    ：param X：（S，S）矩阵，单个单元格等于1
    ：param dx：沿x移动
    ：param dy：沿y移动
    ：return：大小（S，S）的转换矩阵
    """
    x, y, = find_x_y(X)
    xmax, ymax = X.shape
    P = np.zeros(X.shape)
    # Making sure that the agent does not go out of bounds.
    nx = max(0, min(xmax-1, x+dx))
    ny = max(0, min(ymax-1, y+dy))
    P[nx, ny] = 1.
    return P


def get_T_R(grid_size, goal_cell):
    """
    It takes a grid size and a goal cell, and returns the transition matrix and reward matrix for the gridworld

    :param grid_size: the size of the grid
    :param goal_cell: the goal cell
    :return: T is a 4x(grid_size^2)x(grid_size^2) matrix, where each row is a transition matrix for a given action.
    R is a (grid_size^2)x4 matrix, where each row is a reward vector for a given state.
    """
    """
    它获取网格大小和目标单元格，并返回网格世界的转换矩阵和奖励矩阵
    ：param grid_size：网格的大小
    ：param goal_cell：目标单元格
    ：return:T是一个4x（网格大小^2）x（网格尺寸^2）矩阵，其中每一行都是给定动作的转换矩阵。
    R是（grid_size^2）x4矩阵，其中每一行是给定状态的奖励向量。
    """
    T = np.zeros((4, grid_size**2, grid_size**2))
    R = np.zeros((grid_size, grid_size, 4))
    # Unpacking the tuple `goal_cell` into two variables `g_x` and `g_y`.
    # 将元组“goal_cell”解压缩为两个变量“g_x”和“g_y”。
    g_x, g_y = goal_cell

    # A, X, Y, X, Y = P
    # A dictionary that maps the action to the movement along the x and y axis.
    # 将动作映射到沿x和y轴移动的字典。
    action_dx_dy = {
        0 : (0, 1),  # right
        1 : (0, -1), # left
        2 : (-1, 0), # up
        3 : (1, 0)   # down
    }
    for i in range(4):
        dx, dy = action_dx_dy[i]
        for j in range(grid_size**2):
            s = get_one_hot(grid_size**2, j, reshape=(grid_size, grid_size))
            #计算给定状态和动作的转换矩阵。
            Pa = get_P_A(s, dx, dy)
            x, y = find_x_y(s)
            dest_x, dest_y = find_x_y(Pa)
            if dest_x == g_x and dest_y == g_y:
                R[x, y, i] = 1
            if dest_x == x and dest_y == y:
                R[x, y, i] = -1
            R[x, y, i] -= 0.02

            T[i, j, :] = Pa.flatten()

    R = R.reshape(-1, 4)

    return T, R


def get_V(pi, reshape=None):
    v = np.array(pi.V)
    if reshape is not None:
        v = v.reshape(reshape)
    return v

def compute_Pa(Q, beta=1.):
    """
    It computes the probability of each action given the Q-values

    :param Q: the Q-values, a matrix of shape (n_states, n_actions)
    :param beta: inverse temperature
    :return: The probability of each action given the state.
    """
    pi = np.exp(beta * Q)
    pi /= np.sum(pi, 1, keepdims=True)
    return pi

def get_P_a(Q, beta=1.):
    """
    它采用Q值矩阵并返回概率矩阵
    ：param Q：Q值，即状态动作对的值
    ：param beta：softmax的反向温度
    ：return：在状态s中执行动作a的概率。
    """
    #归一化
    Q = np.exp(beta*Q)
    Q /= np.sum(Q,1, keepdims=True)
    return Q

get_det_policy = lambda Q, size: np.argmax(Q, 1).reshape(size, size)


def save_q(id, Q, discount):
    """
    It saves the Q-table to a file

    :param id: a unique identifier for the Q table
    :param Q: a dictionary mapping state -> action values
    :param discount: the discount factor, which is the probability that the agent will continue to the next state
    """
    """
    它将Q表保存到文件中
    ：param id：Q表的唯一标识符
    ：param Q：字典映射状态->操作值
    ：param discount：折扣系数，即代理将继续进入下一状态的概率
    """
    filename = f'q/{id}_{discount}'
    np.save(filename, np.array(Q))

def load_q(id, discount):
    filename = f'q/{id}_{discount}.npy'
    return np.load(filename)


def train_policy(grid_size, goal_cell, n_iter, discount):
    """
    It trains a policy using Q-learning

    :param grid_size: the size of the grid
    :param goal_cell: the cell that the agent is trying to reach
    :param n_iter: number of iterations to run the algorithm for
    :param discount: the discount factor, which is a number between 0 and 1
    :return: The Q-table
    """
    """
    它使用Q-learning培训政策
    ：param grid_size：网格的大小
    ：param goal_cell：代理试图到达的单元格
    ：param n_iter：运行算法的迭代次数
    ：param discount：折扣系数，是介于0和1之间的数字
    ：return：Q表
    """
    T, R = get_T_R(grid_size, goal_cell=goal_cell)
    pi = mdp.QLearning(T, R, discount, n_iter=n_iter, learning_rate=0.1, reset_every=10, policy_epsilon=.4)
    pi.run()
    return pi.Q

def make_policies_q(goals, size, n_iter, discount_factors=(0.8, 0.85, 0.9, 0.95)):
    """
    It trains a policy for each goal, for each discount factor, and saves the resulting Q-table

    :param goals: the list of goals to train the policies for
    :param size: the size of the gridworld
    :param n_iter: number of iterations to train the policy
    :param discount_factors: The discount factor is a number between 0 and 1 that determines how much the agent values
    future rewards
    它为每个目标、每个折扣系数训练策略，并保存生成的Q表
    ：param goals：培训策略的目标列表
    ：param size：网格世界的大小
    ：param n_iter：训练策略的迭代次数
    ：param discount_factors：折扣系数是一个介于0和1之间的数字，用于确定代理的价值
    未来奖励
    """

    results = []
    for df in discount_factors:
        print(f'-------- Make discount factor {df} --------')
        # Using the ThreadPoolExecutor to train the policies in parallel.
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Training a policy for each goal, for each discount factor, and saves the resulting Q-table
            pi = [executor.submit(train_policy, size, g, n_iter, df) for g in goals] ##为2个目标、0.9折扣系数培训策略，并保存生成的Q表
            results += [[df,] + pi,]

    for r in results:
        df = r[0]
        policies = r[1:]
        for i, p in enumerate(policies):
            save_q(i, p.result(), df)




def load_policies_q(discount_factors=(0.8, 0.85, 0.9, 0.95)):
    """
    它加载给定折扣系数的所有Q表
    ：param discount_factors：Q学习算法中使用的折扣因子
    ：return：字典字典。外部字典有作为折扣因子的键，其值为
    字典列表。每个内部字典都有键，键是状态，值是操作。
    """
    res = {}
    # print(discount_factors)
    for df in discount_factors:
        i = 0
        policies = []
        while True:
            try:
                policies.append(load_q(i, df))
            except FileNotFoundError as e:
                break
            i += 1
        res[df] = policies

    return res



def cross_entropy(pi_id, policies, beta=1.5):

    # 代理的策略和其他策略之间的交叉熵
    res = {}
    for df in policies.keys():
        q = policies[df]
        # 计算每个状态的每个动作的概率。
        p = np.vstack([np.expand_dims(get_P_a(i, beta=beta), 0) for i in q])
        #计算代理的策略和其他代理的策略之间的交叉熵。
        res[df] = -np.log(p[pi_id, ...])  + np.log(np.mean(p, 0)) - np.log(1/p.shape[0])
    return res


#    0  right
#    1  left
#    2  up
#    3  down

def print_policy(p):
    from collections import defaultdict
    to_str = lambda i: defaultdict(lambda : 'X', {0 : 'R', 1: 'L', 2: 'U', 3: 'D'})[i]

    for row in p:
        for cell in row:
            print(to_str(cell)+'  ', end='')
        print()




# The goal cell for the two agents.
goals = [(0,19), (0,0)]
size=20


if __name__ == '__main__':

    # It sets the print options for numpy，
    # precision=3： 保留三位小数，
    # suppress = True： 不用科学计数法
    np.set_printoptions(suppress=True, precision=3)
    n_iter=700000
    # It trains a policy for each goal, for each discount factor, and saves the resulting Q-table
    # 它为goals、0.9折扣系数训练策略，并保存生成的Q表
    make_policies_q(goals, size, n_iter, discount_factors=(.9,))
    # Use simulator.py instead
    policies_q = load_policies_q(discount_factors=(.9,))
    # print(policies_q)
    beta=1
    ce_0 = cross_entropy(0, policies_q, beta=beta)
    ce_1 = cross_entropy(1, policies_q, beta=beta)
    #ce_2 = cross_entropy(2, policies_q, beta=beta)

    for df, reg_factor in itertools.product(policies_q.keys(), (1., .9, .7, .5, .3)):
        pi0, pi1 = policies_q[df]
        reg_0 = ce_0[df]
        reg_1 = ce_1[df]
        #reg_2 = ce_2[df]

        p_a_0 = get_P_a((1-reg_factor) * pi0 - reg_factor * reg_0)
        p_a_1 = get_P_a((1-reg_factor) * pi1 - reg_factor * reg_1)
        #p_a_2 = get_P_a((1-reg_factor) * pi2 - reg_factor * reg_2)

        pi  = [pi0, pi1,]
        p_a = [p_a_0, p_a_1,]

        print(f'-------- Discount factor {df} Regul factor {reg_factor} --------')
        i = 1
        P0 = get_det_policy(pi[i], size)
        P0[goals[i][0], goals[i][1]] = 10
        # print_policy(P0)
        print()
        P_A = get_det_policy(p_a[i], size)
        P_A[goals[i][0], goals[i][1]] = 10
        # print_policy(P_A)
        print()
        print()
