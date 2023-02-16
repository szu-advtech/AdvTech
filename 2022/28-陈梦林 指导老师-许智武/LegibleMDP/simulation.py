import numpy as np
"""
它需要一个状态和若干步骤来向前看，并从该状态返回最佳路径的预期熵
：param sim：具有正常策略的模拟器
：param i：起始状态
：param g：目标状态
：return：从当前状态达到目标状态的概率。
"""

#    0  right
#    1  left
#    2  up
#    3  down

class Simulator:

    def __init__(self, T, Q, policy):
        self.T = T
        self.Q = Q
        self.policy = policy

        self.s = None

    @property
    def size(self):
        return self.T.shape[:2]

    @property
    def S(self):
        A_size, S_size = self.size
        S = np.zeros(S_size)
        S[self.s] = 1
        return S


    def initialize(self, s):
        """

        :param s: Integer. State enumeration num
        :return:
        """
        self.s = s



    def step(self):
        """
        > Given the current state, choose an action according to the policy, then update the state according to the
        transition probabilities
        """
        assert self.s is not None, 'call initialize() before step()'
        s = self.s
        a = self.policy(self.Q, s)#随机选取转换矩阵中概率较大的动作
        t = self.T[a, s,:]
        A_size, S_size = self.size
        self.s = np.random.choice(S_size, p=t)


    def make_trajectory_stat(self, s, goal_s, n=100, max_steps=1000):
        """
        >函数“make_trajectory_stat”接受开始状态“s”、目标状态“goal_s”和多个
        轨迹“n”，并返回大小为“n”乘“size”的矩阵，其中每行是轨迹，每列是
        状态
        ：param s：启动状态
        ：param goal_s：要达到的状态
        ：param n：要生成的轨迹数，默认为100（可选）
        ：param max_steps：每个轨迹中要执行的最大步数，默认为1000（可选）
        ：return：结果是大小（n，size）的矩阵，其中n是轨迹的数量，size是状态数
        """
        _, size = self.size
        result = np.zeros((n, size))
        for i in range(n):
            self.initialize(s)
            n_steps = 0
            while True:
                self.step()
                result[i, self.s] += 1
                n_steps += 1
                if (s == goal_s) or (n_steps >= max_steps):
                    break
            print('.', end='')
        print()
        return result

    """
    >函数“softmax_policy”返回一个接受Q表和状态并返回操作的函数
    ：param a：softmax参数，默认为1（可选）
    ：return：接受Q和s的函数，并返回接受Q和i的函数
    接受Q、i和b并返回一个操作。
    """
    @staticmethod
    def softmax_policy(a=1):
        def policy(Q, i, b):
            Q = Q[i, :]
            Q = Q.flatten()
            P = np.exp(b *Q)
            P /= np.sum(P)
            a = np.random.choice(len(Q), p=P)
            return a

        return lambda Q, s: policy(Q, s, a)

def get_trajectory(sim, i, g):
    """
    它接受模拟、状态和目标，并返回从该状态到目标的平均轨迹
    ：param sim：模拟对象
    ：param i：初始状态
    ：param g：目标状态
    ：return：在给定时间处于给定状态的概率。
    """
    traj = sim.make_trajectory_stat(i, g, max_steps=30, n=10)
    traj[traj > 1] = 1
    avg_T = np.sum(traj, 0)
    avg_T /= np.sum(avg_T)
    T = avg_T.reshape((size, size))
    return T

def get_goal_state(size, goals, trg_goal):
    """
    它获取网格的大小、目标列表和目标，并返回目标状态
    ：param size：网格世界的大小
    ：param goals：元组列表，每个元组都是目标位置
    ：param trg_goal：我们想要达到的目标
    ：return：正在返回目标状态。
    """
    # print(size)
    grid = np.zeros((size, size))
    grid[goals[trg_goal][1], goals[trg_goal][0]] = 1

    G = grid.reshape(-1)

    # print("np.argwhere(G)",np.argwhere(G))
    g = np.argwhere(G).reshape(-1)
    return g


def softmax(Q, beta):
    """
    它获取Q值向量和参数β，并返回概率向量
    ：param Q：每个状态动作对的Q值
    ：param beta：逆温度参数
    ：return：正在返回softmax函数。
    """
    Q = np.exp(beta * Q)
    Q /= np.sum(Q, 1, keepdims=True)
    return Q


def tree_search_a(T, entropies, s, n):
    """
    它采用一个转换矩阵、一个熵列表、一个状态和若干步骤，并返回
    给定步骤数后的状态
    ：param T：转换矩阵
    ：param entropies：每个状态的熵列表
    ：param s：我们所处的状态
    ：param n：要执行的步骤数
    ：return：当前状态的最佳交叉熵
    """
    best_children = []
    # print(T.shape[0]) T是当前状态执行当前动作转移到下一个状态的概率矩阵
    for a in range(T.shape[0]):
        #当前状态，当前动作的状态转移矩阵即动作向右，状态399的概率转换列表
        ps_new = T[a, s, ...]
        #选择应该概率最大的状态 即选择当前399状态下向右动作概率最大的状态（399，向右的下一个动作还是399）
        s_new = np.random.choice(T.shape[-1], p=ps_new)
        # print( "tree",tree_search(T, entropies, s_new, n),)
        best_children += tree_search(T, entropies, s_new, n),
    return np.array(best_children)



def tree_search(T, entropies, s, n):
    """
    >对于每个操作，我们采取该操作，然后递归搜索下一步的最佳操作
    ：param T：转换矩阵
    ：param熵：形状矩阵（n_states，n_actions）
    ：param s：我们当前所处的状态
    ：param n：向前看的步骤数
    ：return：接下来n步的最佳CE。
    """
    #交叉熵
    ce = entropies
    #当前状态的最小交叉熵
    best_ce = np.min(ce[s,...])
    if n == 0:
        return best_ce
    h = []#获取当前状态所有动作的交叉熵
    for i in range(T.shape[0]):
        #当前状态，当前动作的状态转移矩阵
        ps_new = T[i, s, ...]
        #下一个状态
        s_new = np.random.choice(T.shape[-1], p=ps_new)

        hc = tree_search(T, entropies, s_new, 0)
        h += hc,
    i_best = np.argmin(h)#选择当前状态的最小交叉熵
    ps_new = T[i_best, s, ...] #通过当前状态的最小交叉熵选择动作
    s_new = np.random.choice(T.shape[-1], p=ps_new)
    ce_best_ch = tree_search(T, entropies, s_new, n - 1)
    return best_ce + ce_best_ch



if __name__ == '__main__':
    from main import get_T_R, load_policies_q, cross_entropy, goals, size
    import matplotlib.pyplot as plt
    # The inverse temperature parameter.
    beta = 20
    # The inverse temperature parameter for the cross entropy.
    beta_ce = 3
    reg_factor = 1
    trg_goal = 0
    discount_factor = 0.9
    """
        它加载给定折扣系数的所有Q表
        ：param discount_factors：Q学习算法中使用的折扣因子
        ：return：字典字典。外部字典有作为折扣因子的键，其值为
        字典列表。每个内部字典都有键，键是状态，值是操作。
        """
    policies_q = load_policies_q(discount_factors=(discount_factor,))
    # print(policies_q)
    """
        它计算代理的策略和其他代理的策略之间的交叉熵
        ：param pi_id：我们要评估的策略的索引
        ：param policies：数据帧字典，每个数据帧都是一个策略
        ：param beta：反温度参数
        ：return：代理的策略和其他策略之间的交叉熵。
    """
    # print(len(goals))
    ce1=[cross_entropy(i, policies_q, beta=beta_ce)for i in range(len(goals))]
    ce = [cross_entropy(i, policies_q, beta=beta_ce)[discount_factor] for i in range(len(goals))]
    Q = policies_q[discount_factor]
    T, R = get_T_R(size, goal_cell=goals[trg_goal])




    def regularized_policy(Q, s):
        """
        >该函数将Q表和当前状态作为输入，并返回一个动作作为输出
        ：param Q：Q表
        ：param s：当前状态
        ：return：返回的操作是代理正在执行的操作。
        """
        # print("s",s)
        n = 10
        # print(ce[trg_goal])
        #递归搜索当前状态接下来10步各动作的最小交叉熵
        # Calculating the best CE for the next n steps.
        a = tree_search_a(T, ce[trg_goal], s, n)
        # print(np.mean(a))
        a -= np.mean(a)
        # 计算在当前状态下执行每个操作的概率。
        proba_base = softmax(Q, beta_ce)
        h_a =np.log(proba_base)
        # The regularized Q-table.
        Q_reg = h_a - reg_factor * a
        pi = softmax(Q_reg, beta)
        pi_sel = pi
        Pa = pi_sel[s,:]
        a = np.random.choice(len(Pa), p=Pa)
        return a
    sim     = Simulator(T, Q[trg_goal], Simulator.softmax_policy(a=beta))
    sim_reg = Simulator(T, Q[trg_goal], regularized_policy)
    # It takes the size of the grid, the list of goals, and the target goal, and returns the goal state
    # 它获取网格的大小、目标列表和目标，并返回目标状态
    g = get_goal_state(size, goals, trg_goal)
    print("g",g)

    # Creating a figure with two subplots.
    ##创建具有两个子地块的地物。
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
    ax1.set_title('Normal')
    ax2.set_title('Regularized')
    # The starting state.
    s = size ** 2 - 48
    # Plotting the trajectory of the agent.
    for i in range(size**2-1, 0, -5): ##
        s = i
        """
        它接受模拟、状态和目标，并返回从该状态到目标的平均轨迹
        ：param sim：模拟对象plt.subplots(nrows=1, ncols=2, figsize=(6, 3))
        ：param i：初始状态
        ：param g：目标状态
        ：return：在给定时间处于给定状态的概率。
        """
        # Getting the trajectory of the agent.
        traj = get_trajectory(sim, s, g)

        # Getting the trajectory of the agent.
        traj_reg = get_trajectory(sim_reg, s, g)
        # print("traj",traj)
        # print("traj_reg", traj_reg)
        # Plotting the trajectory of the agent.
        ax1.imshow(traj)
        ax2.imshow(traj_reg)
        plt.pause(1)