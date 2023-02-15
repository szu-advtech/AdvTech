import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix


class Reservior:
    """
    class maintains a sketch of the dynamic graph
    """
    def __init__(self, edges, vertices, dim=10, seed=24):
        self.reservior = {}
        self.degree = {}          # 顶点的度
        self.init_edges = edges
        self.vertices = vertices
        self.reservior_dim = dim  # 蓄水池单条记录的维度
        self.seed = seed
        self.__build()

    def __build(self):
        """
        construct initial reservior using the inital graph (edge list)
        :return:
        """
        # 根据初始边和节点初始化蓄水池
        g = nx.Graph()    # 创建一个图对象
        g.add_edges_from(self.init_edges)  # 添加初始边进入图对象
        for v in self.vertices:
            if v in g:
                nbrs = list(g.neighbors(v))     # 生成邻居节点列表
                np.random.seed(self.seed)
                indices = np.random.randint(len(nbrs), size=self.reservior_dim)  # 随机初始化邻居节点个数的单条储水池记录
                self.reservior[v] = np.array([nbrs[idx] for idx in indices])     # 生成节点v的蓄水池记录
                self.degree[v] = len(nbrs)
            else:
                self.reservior[v] = np.array([None] * self.reservior_dim)        # 新节点进入
                self.degree[v] = 0

    def update(self, edges):
        """
        update the reservior based on the incoming edge(s)
        :param edges: new incoming edges
        :return:
        """
        assert len(edges)  # assert（断言）用于判断一个表达式，在表达式条件为 false 的时候触发异常。
        for edge in tqdm(edges):
            u, v = edge

            # update u's reservior, edges can not be duplicated in the training and updating
            assert v not in self.reservior[u]

            self.degree[u] += 1
            indices = np.random.randint(self.degree[u], size=self.reservior_dim)
            replace_idx = np.where(indices == self.degree[u] - 1)
            self.reservior[u][replace_idx] = v

            # update v's reservior, edges can not be duplicated in the training and updating
            assert u not in self.reservior[v]  # 不可重复更新或训练同一条边  如果u，v已存在，则返回Assertion错误

            self.degree[v] += 1
            indices = np.random.randint(self.degree[v], size=self.reservior_dim)
            replace_idx = np.where(indices == self.degree[v] - 1)  # 返回替换位置的idx
            # np.where(condition) 当where内只有一个参数时，那个参数表示条件
            # 当条件成立时，where返回的是每个符合condition条件元素的坐标,返回的是以元组的形式
            self.reservior[v][replace_idx] = u


class WalkUpdate:
    """
    WalkUpdate update the Reservior and generate new batch of walks.
    """
    # 根据原始图结构，创建初始蓄水池和初始随机游走
    def __init__(self, init_edges, vertices, walk_len=3, walk_per_node=5, prev_percent=1, seed=24):
        self.init_edges = init_edges
        self.walk_len = walk_len
        self.walk_per_node = walk_per_node
        self.prev_percent = prev_percent   # 新生成的游走与旧游走进行混合产生新的训练集的比例
        self.seed = seed
        self.reservior = Reservior(edges=self.init_edges, vertices=vertices)  # 初始化蓄水池
        # previous walks
        self.prev_walks = self.__init_walks()
        # new generated walks
        self.new_walks = None
        # training walks = new walks + "percent" * old walks
        self.training_walks = None

    # 初始随机生成的游走集(不使用储水池)
    def __init_walks(self):
        """
        Initial walks generated from random walking
        :return:
        """
        g = nx.Graph()
        g.add_edges_from(self.init_edges)
        rand = random.Random(self.seed)
        walks = []
        nodes = list(g.nodes())
        for cnt in range(self.walk_per_node):  # 每个节点生成walk_per_node个游走
            rand.shuffle(nodes)                # 打乱节点数组
            for node in nodes:
                walks.append(self.__random_walk(g, node, rand=rand))
        return walks

    # 生成单个随机游走
    def __random_walk(self, g, start, alpha=0, rand=random.Random(0)):
        """
        return a truncated random walk
        :param alpha: probability of restars
        :param start: the start node of the random walk
        :return:
        """
        walk = [start]
        while len(walk) < self.walk_len:
            cur = walk[-1]
            if len(list(g.neighbors(cur))) > 0:
                if rand.random() >= alpha:
                    walk.append(rand.choice(list(g.neighbors(cur))))
                else:
                    walk.append(walk[0])
            else:
                break
        return walk

    # 动态过程中产生新边，根据新边的节点结合储水池产生随机游走
    def __generate(self, new_edges, update_type="randomwalk"):
        walk_set = []
        rand = random.Random(self.seed)
        # using random walk in the reservior for updating new set of walks for training
        # it's slower but very accurate, it is probabilistically equal to do random walk
        # in the whole graph with all edges so far arrived
        if update_type == "randomwalk":
            start_node = []
            for edge in new_edges:
                u, v = edge
                start_node.append(u)
                start_node.append(v)
            for n in set(start_node):
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[n])
                    y = rand.choice(self.reservior.reservior[x])
                    walk_set.append([n, x, y])

            self.new_walks = walk_set
            old_samples = [w for w in self.prev_walks if w[0] not in start_node]
            # 生成新的训练样本
            self.training_walks = self.new_walks + old_samples
            print("length of training walks: %d" % len(self.training_walks))
            self.prev_walks = walk_set + old_samples

            return

        # current implementation works for len = 3 and 4
        assert self.walk_len < 5

        # decide the number of walk per node???
        for edge in new_edges:
            u, v = edge
            if self.walk_len == 3:
                walk_set = [x for x in walk_set if x[0] != v]
                walk_set = [x for x in walk_set if x[0] != u]

                # walk u-v-x
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[v])
                    walk_set.append([u, v, x])
                # walk v-u-x
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[u])
                    walk_set.append([v, u, x])

                self.prev_walks = [x for x in self.prev_walks if x[0] != v]
                self.prev_walks = [x for x in self.prev_walks if x[0] != u]
            elif self.walk_len == 4:
                # walk u-v-xy
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[v])
                    y = rand.choice(self.reservior.reservior[x])
                    walk_set.append([u, v, x, y])

                # walk v-u-xy
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[u])
                    y = rand.choice(self.reservior.reservior[x])
                    walk_set.append([v, u, x, y])

                # walk x-u-v-y
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[u])
                    y = rand.choice(self.reservior.reservior[v])
                    walk_set.append([x, u, v, y])

                # walk x-v-u-y
                for cnt in range(self.walk_per_node):
                    x = rand.choice(self.reservior.reservior[v])
                    y = rand.choice(self.reservior.reservior[u])
                    walk_set.append([x, v, u, y])
            else:
                pass

        self.new_walks = walk_set
        old_samples = self.prev_walks
        # random.sample(self.prev_walks, int(self.prev_percent * len(self.prev_walks)))
        self.training_walks = self.new_walks + old_samples
        self.prev_walks = self.training_walks
        # self.prev_walks.extend(walk_set)

    #  针对新到达的边，更新蓄水池以及随机游走集
    def update(self, new_edges):
        """
        Updating reservior and generate new set of walks for re-training using newly come edges
        :param new_edges: newly arrived edges
        :return: new set of training walks
        """
        # update reservior
        self.reservior.update(new_edges)
        # if reconduct randomwalk then the new set of walks are probabilistically equal to conducting
        # randomwalk in the graph with all edges so far, it's slower than approximated method
        self.__generate(new_edges, update_type="randomwalk")
        return self.training_walks


class NetWalk_update:
    """
    Preparing both training initial graph walks and testing list of walks in each snapshot
    """
    def __init__(self, path, walk_per_node=5, walk_len=3, init_percent=0.5, snap=10):
        """
        Initialization of data preparing
        :param path: Could be either edge list file path, or the tuple including training and testing edge lists
        :param walk_per_node: for each node, how many walks start from the node to be sampled
        :param walk_len: the length of walk
        :param init_percent: initial percentage of edges sampled for training
        :param snap: number of edges in each time snapshot for updating
        """
        self.data_path = path
        self.walk_len = walk_len
        self.init_percent = init_percent
        self.snap = snap
        self.vertices = None
        self.idx = 0
        self.walk_per_node = walk_per_node

        if isinstance(path, str):
            self.data = self.__get_data()
        else:
            test = path[0][:, 0:2]
            train = path[1]
            self.data = self.__get_data_mat(test, train)

        init_edges, snapshots = self.data
        self.walk_update = WalkUpdate(init_edges, self.vertices, walk_len=self.walk_len, walk_per_node=self.walk_per_node, prev_percent=1, seed=24)

    def __get_data_mat(self, test, train):
        """
        Generate initial walk list for training and list of walk lists in each upcoming snapshot
        :param test: edge list for testing
        :param train: edge list for training
        :return: initial walk list for training and list of walk lists in each upcoming snapshot
        """
        edges = np.concatenate((train, test), axis=0)  # 拼接训练集和测试集数组
        self.vertices = np.unique(edges)  # 从边集中返回顶点
        init_edges = train
        print("total edges: %d \ninitial edges: %d \ntotal vertices: %d\n"
              % (len(edges), len(init_edges), len(self.vertices)))

        snapshots = []
        current = 0
        while current < len(test):
            # if index >= len(edges), equals to edges[current:]
            # length of last snapshot <= self.snap
            snapshots.append(test[current:current + self.snap])
            current += self.snap
        print("number of snapshots: %d \nedges in each snapshot: %d\n" % (len(snapshots), self.snap))
        data = (init_edges, snapshots)
        return data

    def __get_data(self):
        """
        Generate initial walk list for training and list of walk lists in each upcoming snapshot
        :return: initial walk list for training and list of walk lists in each upcoming snapshot
        """
        edges = np.loadtxt(self.data_path, dtype=int, comments='%')
        self.vertices = np.unique(edges)
        init_idx = int(len(edges) * self.init_percent)
        init_edges = edges[:init_idx]
        print("total edges: %d \ninitial edges: %d \ntotal vertices: %d\n"
              % (len(edges), len(init_edges), len(self.vertices)))

        snapshots = []
        current = init_idx
        while current < len(edges):
            # if index >= len(edges), equals to edges[current:]
            # length of last snapshot <= self.snap
            snapshots.append(edges[current:current + self.snap])
            current += self.snap
        print("number of snapshots: %d \nedges in each snapshot: %d\n" % (len(snapshots), self.snap))
        data = (init_edges, snapshots)
        return data

    def run(self):
        """
        perform netwalk program with input data
        :return:
        """
        init_edges, snapshots = self.data
        walk_update = WalkUpdate(init_edges, self.vertices, self.walk_per_node)
        # call netwalk core
        # save embedding
        # save model
        for snapshot in snapshots:
            training_set = walk_update.update(snapshot)

            self.getOnehot(training_set)
            print(training_set)
            # load model
            # update embeddings

    def getNumsnapshots(self):
        """
        get number of total time snapshots
        :return: The number of total time snapshots
        """
        init_edges, snapshots = self.data
        return len(snapshots)

    def nextOnehotWalks(self):
        """
        get next list of walks for re-training in next time snapshot
        :return: next list of walks for re-training in next time snapshot
        """
        if not self.hasNext():
            return False
        _, snapshots = self.data
        snapshot = snapshots[self.idx]
        self.idx += 1
        training_set = self.walk_update.update(snapshot)

        return self.getOnehot(training_set)

    def hasNext(self):
        """
        Checking if still has next snapshot
        :return: true if has, or return false
        """
        init_edges, snapshots = self.data
        if self.idx >= len(snapshots):
            return False
        else:
            return True

    def getInitWalk(self):
        """
        Get inital walk list
        :return: list of walks for initial training
        """
        walks = self.walk_update.prev_walks
        return self.getOnehot(walks)


    def getOnehot(self, walks):
        """
        transform walks with id number to one-hot walk list
        :param walks: walk list
        :return: one-hot walk list
        """
        walk_mat = np.array(walks, dtype=int)-1
        rows = walk_mat.flatten()
        cols = np.array(range(len(rows)))
        print(len(rows))
        data = np.array([1] * (len(rows)))
        coo = coo_matrix((data, (rows, cols)), shape=(len(self.vertices), len(rows)))
        onehot_walks = csr_matrix(coo)
        return onehot_walks.toarray()


# demo with dataset karate and emails
# data without weight, undirected graph, no duplicate edges
if __name__ == "__main__":
    data_path = "../../data/karate.edges"
    #data_path = "../../data/cit-DBLP.edges.edges"
    # ""../../data/karate/karate.edgelist"
    netwalk = NetWalk_update(data_path)
    netwalk.run()
