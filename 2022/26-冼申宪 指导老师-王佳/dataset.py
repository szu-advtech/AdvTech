import utils
from utils import args, tqdm
import numpy as np

data_home = './data'


class DatasetReader:
    def __init__(self, ds):
        self.ds = ds
        self.N = 8058436
        self.reader = self.yc  # 把迭代函数存到变量里

    def yc(self, frac=1):
        pbar = tqdm(desc='read data', total=self.N)
        f = open(f'{data_home}/ub.txt', 'r')
        for line in f:
            pbar.update(1)
            line = line[:-1]  # 去掉换行符
            sid, vid_list_str = line.split()
            vid_list = []
            for vid in vid_list_str.split(','):  # 循环每个(vid,type)二元组--他这里偷懒使用vid表示整个三元组
                vid, cls ,ts= vid.split(':')
                vid = int(vid)
                cls = int(cls)  # cls: 0, 1, 2, ... 行为类型
                vid_list.append([vid, cls])
            yield vid_list
        f.close()
        pbar.close()


class DataProcess:
    def __init__(self, ds, adj_length, seq_length):
        self.ds = ds

        self.vid2node = {}      # vid转节点id 字典
        self.vid2node['[MASK]'] = 0

        self.DR = DatasetReader(ds)     # 设定数据迭代器
        self.G_in, self.G_out, self.train_data, self.test_data = self.build_graph(seq_length)

        rdm = np.random.RandomState(777)
        rdm.shuffle(self.train_data)

        rdm = np.random.RandomState(333)
        rdm.shuffle(self.test_data)

        args.update(nb_nodes=len(self.vid2node))
        args.update(nb_edges_0=self.G_in[0].get_nb_edges())
        args.update(nb_edges_1=self.G_in[1].get_nb_edges())

        self.adj_in_0 = self.build_adj(self.G_in[0], adj_length)
        self.adj_out_0 = self.build_adj(self.G_out[0], adj_length)
        self.adj_in_1 = self.build_adj(self.G_in[1], adj_length)
        self.adj_out_1 = self.build_adj(self.G_out[1], adj_length)

        self.adjs_tmp = [self.adj_in_0, self.adj_out_0, self.adj_in_1, self.adj_out_1]
        self.adjs = [a[0] for a in self.adjs_tmp]

    def tos(self,df, test_data):
        click_history = list(df[df['behavior'] == 0]['Item'].unique())
        share_history = list(df[df['behavior'] == 1]['Item'].unique())
        test_data.append([share_history[:-1], click_history, share_history[-1]])



    # 相对位置划分
    def build_graph(self, seq_length):
        G_in = [utils.Graph() for i in range(2)]    #返回的是空白图
        G_out = [utils.Graph() for i in range(2)]   #返回的是空白图
        train_data = []
        test_data = []
        # 以下针对每一个会话
        for num_data, vid_list in enumerate(self.DR.reader()):  # 迭代器取值,第一个是序号下标

            vid_list_for_graph = [[] for i in range(2)]  # 存储当前会话需要用于构图的vid
            vid_list_for_train = [[] for i in range(2)]
            first_pos = [{} for i in range(2)]
            buy_num = 0
            buy_now = 0
            for i, (vid, typ) in enumerate(vid_list):
                typ=1-typ
                if(typ==1):
                    buy_num += 1
            # 以下针对一个会话中的每一个行为
            for i, (vid, typ) in enumerate(vid_list):
                typ = 1-typ
                if vid not in self.vid2node:
                    self.vid2node[vid] = len(self.vid2node)  # 分配序号？

                if typ==1 :     # 记录这是该序列中第几个购买行为
                    buy_now += 1

                for_train = True
                if (buy_num>=3 and buy_num-buy_now==0):     # 最后一个购买行为作为测试集
                    for_train = False

                if for_train:
                    vid_list_for_graph[typ].append(vid)  # 把该vid丢进队列准备构图


                if typ == 1 and vid not in first_pos[1]:  # vid在该会话中是目标行为且第一次出现
                    share_history = vid_list_for_train[1]  # 复制该行为之前的目标行为序列
                    if vid not in first_pos[0]:  # vid在之前也没有点击过
                        click_history = vid_list_for_train[0]
                    else:
                        k = first_pos[0][vid]
                        click_history = vid_list_for_train[0][:k]  # 如果点击过，那就将点击之前的信息作为历史信息

                    if len(click_history) >= 5 and len(share_history) >= 1:  # 辅助行为长度≥5 且 目标行为历史≥1 时才构造

                        # share_history = torch.tensor(share_history)
                        # click_history = torch.tensor(click_history)
                        # vid = torch.tensor(vid)
                        seq_share = [share_history[-seq_length:], click_history[-seq_length:], vid]
                        # seq_share = [share_history, click_history, vid]  # 构造模型输入的元数据
                        # seq_share = torch.tensor(seq_share)
                        if for_train:  # 根据目标行为区分元数据属于训练集还是测试集
                            train_data.append(seq_share)
                        else:
                            test_data.append(seq_share)

                if vid not in first_pos[typ]:
                    first_pos[typ][vid] = len(vid_list_for_train[typ])  # 记录vid在vid_list_for_train首次出现的下标位置
                vid_list_for_train[typ].append(vid)

            for typ in range(2):
                for i, vid in enumerate(vid_list_for_graph[typ]):
                    if i == 0:  # 跳过第一个
                        continue
                    now_node = self.vid2node[vid]
                    pre_node = self.vid2node[vid_list_for_graph[typ][i - 1]]
                    if now_node != pre_node:
                        G_in[typ].add_edge(pre_node, now_node)  # 出入度分别记录
                        G_out[typ].add_edge(now_node, pre_node)
                    else:
                        pass

        # train_data = torch.tensor(train_data)

        return G_in, G_out, train_data, test_data

    def build_adj(self, G, M):
        # M: number of adj per node
        N = args.nb_nodes
        # adj shape: [N, M]
        adj = [None] * N        # N个空序列
        adj[0] = [0] * M

        w = [None] * N
        w[0] = [0] * M

        rdm = np.random.RandomState(555)
        pbar = tqdm(total=N - 1, desc='building adj') #初始化进度条
        for node in range(1, N):
            pbar.update(1)      #进度条赋值
            adj_list = G.get_adj(node)  #从大图中获取该节点的邻接节点集
            if len(adj_list) > M:
                adj_list = rdm.choice(adj_list, size=M, replace=False).tolist()     #超出就随机选一部分
            mask = [0] * (M - len(adj_list))
            adj_list = adj_list[:] + mask   #数量不足用0补充
            adj[node] = adj_list
            w_list = [G.edge_cnt.get((node, x), 0) for x in adj_list]   #node与邻接节点的连接数 node→x 单向？
            w[node] = w_list
        pbar.close()    #关闭进度条
        return [adj, w]     #返回邻接点集adj和连接数w(权重或计数器)





class Data():
    def __init__(self):
        self.dp = DataProcess(args.ds, args.adj_length, args.seq_length)    # 处理数据

        self.adjs = self.dp.adjs
        self.vid2node = self.dp.vid2node
        self.load_data()
        self.status = 'train'
        self.graph = []
        self.transA()

    def load_data(self):
        self.data = self.dp.train_data + self.dp.test_data
        nb_train = len(self.dp.train_data)  # 训练集数量
        nb_non_train = len(self.dp.test_data)  #
        nb_vali = nb_non_train // 3  # 验证集数量
        nb_test = nb_non_train - nb_vali  # 测试集数量

        nb_data = len(self.data)  # 数据总数
        assert nb_data > 0
        args.update(nb_data=nb_data, nb_train=nb_train, nb_vali=nb_vali, nb_test=nb_test)

    def pad_seq(self, node_list):
        L = args.seq_length
        if len(node_list) < L:
            node_list = node_list + [0] * (L - len(node_list))
        return node_list


    def sample_neg(self, pos, rdm):
        """
        采集负样本
        pos: 目标下标
        rdm：随机数采集器
        """
        neg = set()
        while len(neg) < args.num_neg:
            n = rdm.randint(args.nb_nodes)
            if n != 0 and n != pos and n not in neg:
                neg.add(n)
        neg = sorted(neg)
        return neg

    def get_data_by_idx(self, idx, rdm):
        # 读取
        share_history, click_history, pos = self.data[idx]
        # 转换
        pos = self.vid2node[pos]
        share_seq = [self.vid2node[vid] for vid in share_history]
        click_seq = [self.vid2node[vid] for vid in click_history]
        # 填充
        share_list = self.pad_seq(share_seq)
        click_list = self.pad_seq(click_seq)
        #拼接
        ret = [share_list, click_list, pos]

        if self.status == 'train':
            neg = self.sample_neg(pos, rdm)
            ret.append(neg)
        return ret

    def get_batch_by_idxs(self, idxs, rdm=None):
        data = None
        for idx in idxs:
            d = self.get_data_by_idx(idx, rdm)
            n = len(d)
            if data is None:
                data = [[] for _ in range(n)]
            for i in range(n):
                data[i].append(d[i])

        # data: [0-seq, 1-typ, 2-len, 3-nxt, 4-label]
        batch = [np.array(d) for d in data]
        return batch

    def gen_train_batch_for_train(self, batch_size):
        # 返回一个数量为batch_size的数据下标迭代器
        rdm = np.random.RandomState(333)
        while True:
            idxs = list(range(args.nb_train))
            rdm.shuffle(idxs)
            for i in range(0, args.nb_train, batch_size):
                batch = self.get_batch_by_idxs(idxs[i: i + batch_size], rdm)
                yield batch

    def get_data_idxs(self, name):
        """根据名字返回对应部分的下标起始和结束的下标"""
        if name == 'train':
            return 0, args.nb_train
        if name == 'vali':
            return args.nb_train, args.nb_train + args.nb_vali
        if name == 'test':
            return args.nb_train + args.nb_vali, args.nb_data

    def gen_metric_batch(self, name, batch_size):
        self.status = 'metric'
        begin_idx, end_idx = self.get_data_idxs(name)
        yield from self.gen_data_batch(begin_idx, end_idx, batch_size)
        self.status = 'train'

    def gen_all_batch(self, batch_size):
        begin_idx = 0
        end_idx = args.nb_data
        yield from self.gen_data_batch(begin_idx, end_idx, batch_size)

    def gen_data_batch(self, begin_idx, end_idx, batch_size):
        for i in range(begin_idx, end_idx, batch_size):
            a, b = i, min(end_idx, i + batch_size)
            batch = self.get_batch_by_idxs(range(a, b))
            yield batch


    def metric(self, pred_list, true_vid):
        pred_list = np.array(pred_list)
        true_vid = np.expand_dims(np.array(true_vid), -1)
        print(pred_list.shape)
        print(true_vid.shape)

        k = 100
        acc_ar = (pred_list == true_vid)[:, :k]  # [BS, K]
        acc = acc_ar.sum(-1)

        rank = np.argmax(acc_ar[:, :k], -1) + 1
        mrr = (acc / rank).mean()
        ndcg = (acc / np.log2(rank + 1)).mean()

        acc = acc.mean()
        # print(acc_ar)
        # print(mrr)
        # input()
        acc *= 100
        mrr *= 100
        ndcg *= 100
        ret = acc
        return ret, '{:.3f},{:.4f},{:.4f}'.format(acc, mrr, ndcg)

    def transA(self):
        for adj in self.adjs:
            begin = []
            end = []
            for node in range(len(adj)):
                for l in adj[node]:
                    if l != 0:
                        begin.append(node)
                        end.append(l)
            self.graph.append([begin,end])









def main():
    pass


if __name__ == '__main__':
    main()







