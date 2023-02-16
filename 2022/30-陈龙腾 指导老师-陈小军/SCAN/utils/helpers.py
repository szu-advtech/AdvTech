import copy
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import torch.backends.cudnn as cudnn
import torchvision
import scipy
import faiss

from tqdm import tqdm
from torchnet import meter
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, Sampler
from torch.utils.data import Dataset
from models import *
from faiss import normalize_L2
from utils.evaluate_utils import _hungarian_match

class StreamBatchSampler(Sampler):

    def __init__(self, primary_indices, batch_size):
        self.primary_indices = primary_indices
        self.primary_batch_size = batch_size

    def __iter__(self):
        primary_iter = iterate_eternally(self.primary_indices) # 将indices转成迭代的元素
        return (primary_batch for (primary_batch)
                in grouper(primary_iter, self.primary_batch_size)
                ) # 构造batch, 大小加一个迭代的的东西。

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices) # list

    return itertools.chain.from_iterable(infinite_shuffles())


class MixBatchSampler(Sampler):
    def __init__(self, dateset, l_indices, indices, alpha, batch):
        # 获得对应的位置
        self.len = len(dateset)
        self.l_indices = l_indices
        self.indices = indices
        self.batch = batch
        self.alpha = alpha

    def __iter__(self):# 抽样这里需要好好写
        out = [               ]
        s = np.float32(np.random.dirichlet(self.alpha, 1)) # 生成四个和为1 的采样概率
        for index in range(0, len(self.len)):
            nei = np.random.choice(self.l_indices[index], p=s[0])
            print(nei)
        # 迭代的返回对应的batch(近邻，非近邻，强伪，弱伪)
        # 木的目的是从每一个xi构造一个batch


        # nei_num = int(self.batch * (1. - self.R))
        # other_num = self.batch - nei_num
        # for index in self.indices.shape[0]:
        #     neighbor_index = np.random.choice(self.indices[index], nei_num)
        #     for nei in neighbor_index:
        #         out.append(nei)
        #     for i in range(other_num):
        #         j = np.random.choice(range(0, self.indices.shape[0]), 1)[0]
        #         if j != index:
        #             other_index = np.random.choice(self.indices[j], 1)[0]
        #             out.append(other_index)

        primary_iter = iterate_eternally_Mix(self.primary_indices)
        return (out_batch for (out_batch) in grouper(out, self.batch)) # out时一个索引表

        # primary_iter = iterate_eternally(self.primary_indices)
        # return (primary_batch for (primary_batch)
        #         in grouper(primary_iter, self.primary_batch_size)
        #         )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_eternally_Mix(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(indices) # 将多个迭代器连接成一个统一的迭代器的最高效的方法



def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)

def split_batch(init_list, batch_size):
    groups = zip(*(iter(init_list),) * batch_size)
    end_list = [list(i) for i in groups] # 需要去掉尾巴
    count = len(init_list) % batch_size
    # end_list.append(init_list[-count:]) if count != 0 else end_list #这里是将剩余的部分加上
    return end_list


class DataFolder(Dataset):
    def __init__(self, dataset):
        super(DataFolder, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset

        if isinstance(transform, dict):
            self.e_transform = transform['standard']
            self.w_transform = transform['weak']
            self.s_transform = transform['augment']
        else:
            self.e_transform = transform
            self.w_transform = transform
            self.s_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x = self.dataset.__getitem__(index)
        sample = x['image']

        ### If unlabeled grab pseudo-label
        labeled_case = self.is_labeled[index]
        if labeled_case == 0:
            target = self.p_labels[index] # 无标签得到的是最大值
        else:
            target = self.p_labels[index] # 有标签得到的是自己的标签

            ### If in feat mode just give base images
        if self.feat_mode == True:
            aug_images = []
            e_sample = self.e_transform(sample)
            aug_images.append(e_sample)
            return aug_images

        else:
            if labeled_case == 1: # 强伪标签数据使用弱增强
                aug_images = []
                for i in range(self.aug_num):
                    t_sample = self.w_transform(sample)     # try the self.e_transform(sample)
                    aug_images.append(t_sample)
            else:
                aug_images = []
                for i in range(self.aug_num): # 弱伪标签数据使用强增强
                    t_sample = self.s_transform(sample)
                    aug_images.append(t_sample)

            return aug_images, target


class DBSS(DataFolder):
    def __init__(self, args, dataset, labels, l_indices, load_im=False):
        super(DBSS, self).__init__(dataset)

        self.labeled_idx = l_indices.numpy()
        self.unlabeled_idx = np.setdiff1d(np.arange(len(self.dataset)), self.labeled_idx)
        self.targets = np.asarray(self.dataset.targets)
        self.all_labels = []
        self.acc = 0
        self.feat_mode = True
        self.aug_num = args.aug_num
        self.probs_iter = np.zeros((len(self.dataset), len(self.dataset.classes)))
        self.p_labels = []
        self.p_weight = np.ones(len(self.dataset))
        self.label_dis = 1/(len(self.dataset.classes)) * np.ones(len(self.dataset.classes))
        self.k = args.topk_lp
        self.gamma = args.gamma
        self.W = None
        self.relabel_dataset(labels)
        self.mix_batch_size = args.mix_batch_size


    def relabel_dataset(self, labels):
        self.all_labels = copy.copy(np.asarray(self.dataset.targets))
        self.all_labels[self.labeled_idx] = labels.numpy()     # copy the pseudo labels to all_labels
        self.p_labels = np.ones_like(self.all_labels) * - 1 # 返回一个用-1填充的跟输入 形状和类型 一致的数组
        self.is_labeled = np.ones(len(self.dataset))
        self.is_labeled[self.unlabeled_idx] = 0 # 无标签位置标签设置为0，反之


    def get_knn2(self, D, I, k):
        D = torch.from_numpy(D)
        I = torch.from_numpy(I)
        n = len(I)
        I_ = torch.zeros(n, n)
        row_ind = torch.arange(n)
        row_rep = row_ind.view(-1, 1).expand_as(I)
        I_[row_rep, I] = 1
        S = torch.zeros(n, n)
        for i in range(n):
            S[i] = 1/(1+torch.norm(I_[0].unsqueeze(0)-I_, dim=1))
        D2, I2 = torch.topk(S, k)

        return D2, I2


    def one_iter_true(self, X, logit, D, I, max_iter, epoch, num=0.005,  R1=0.8, R2=0.99):
        alpha = 0.99
        labeled_idx = self.labeled_idx
        unlabeled_idx = self.unlabeled_idx
        k = self.k

        # before propagating, get acc of the points of different density, 0.8-0.99
        prob = torch.from_numpy(logit).softmax(-1)
        max_prob, predicted = torch.max(prob, dim=1)  # 最大列概率作为置信度判断标准
        low_index = (max_prob < R1).nonzero().numpy()  # Todo: 这个low/middle/high的阈值R1、R2可以调整, 调整完跑几十个epoch，对照之前的看看效果如何
        high_index = (max_prob > R2).nonzero().numpy()
        middle_index = ((R1 <= max_prob) & (max_prob <= R2)).nonzero().numpy()  # low  R1  middle  R2 high

        low_index = np.setdiff1d(low_index, labeled_idx)  # 有标签的低置信度的位置
        high_index = np.setdiff1d(high_index, labeled_idx)
        middle_index = np.setdiff1d(middle_index, labeled_idx)

        from utils.evaluate_utils import _hungarian_match
        match = _hungarian_match(predicted, self.targets, preds_k=len(self.dataset.classes),
                                 targets_k=len(self.dataset.classes))
        reordered_preds_all = np.zeros_like(predicted)
        for pred_i, target_i in match:
            reordered_preds_all[predicted == int(pred_i)] = int(target_i)
        rp_label_acc = (reordered_preds_all[labeled_idx] == self.targets[labeled_idx]).mean() # 有标签部分
        low_acc = (reordered_preds_all[low_index] == self.targets[low_index]).mean()
        middle_acc = (reordered_preds_all[middle_index] == self.targets[middle_index]).mean()
        if len(high_index) > 0:
            high_acc = (reordered_preds_all[high_index] == self.targets[high_index]).mean()
        else:
            high_acc = 0
        print('before propagating, rp_label_acc:{:.2f}, low_acc:{:.2f}, middle_acc:{:.2f}, high_acc:{:.2f}'.format(
            rp_label_acc, low_acc, middle_acc, high_acc))
        print('before propagating, low_num:{}, middle_num:{}, high_num:{}'.format(len(low_index), len(middle_index),
                                                                                  len(high_index)))

        N = D.shape[0]  # < 50000 + (k+n)

        # Create the graph--Wij, 不平衡矩阵稀疏化处理
        row_idx = np.arange(N)
        D = D[:, 1:] # 排除自己
        I = I[:, 1:]
        row_idx_rep = np.tile(row_idx, (k, 1)).T # 只沿着X轴复制的方法, k + maxLen - 1倍的关系
        indices = row_idx_rep.flatten('F')
        indptr = I.flatten('F')
        W = scipy.sparse.csr_matrix((D.flatten('F'), (indices, indptr)), shape=(N, N)) # scipy.sparse.csr_matrix可以创建一个空的稀疏矩阵, shape这里的行大小很重要
        # 原来的长度 len(row_idx_rep.flatten('F')) = 500000
        # len(D.flatten("F")) = 250000
        # len(I.flatten('F')) = 250000

        # W = torch.tensor(D)
        W = W + W.T.multiply(W.T > W) - W.multiply(W.T > W)

        # Normalize the graph
        W = W - scipy.sparse.diags(W.diagonal())
        S = W.sum(axis=1)
        S[S == 0] = 1
        D = np.array(1. / np.sqrt(S))
        D = scipy.sparse.diags(D.reshape(-1)) # 得到度矩阵
        Wn = D * W * D # 得到归一化矩阵

        # Initiliaze the y vector for each class
        labels = self.all_labels
        Z = np.zeros((N, len(self.dataset.classes)))
        A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn # 对角线上的稀疏矩阵
        # 返回一个稀疏 (m x n) 矩阵，其中第 k 个对角线全为 1，其他全为 0。
        Y = np.zeros((N, len(self.dataset.classes)))
        Y[labeled_idx, labels[labeled_idx]] = 1

        print("标签传播开始")
        for i in range(len(self.dataset.classes)):
            f, _ = scipy.sparse.linalg.cg(A, Y[:, i], tol=1e-3, maxiter=max_iter)
            Z[:, i] = f
        Z[Z < 0] = 0 # 预测矩阵n*c
        print("标签传播结束")
        # Z[5000-n]这些都是对应的结果

        from utils.evaluate_utils import _hungarian_match
        match = _hungarian_match(self.p_labels, self.targets, preds_k=len(self.dataset.classes),
                                 targets_k=len(self.dataset.classes))
        reordered_preds = np.zeros_like(self.p_labels)
        for pred_i, target_i in match:
            reordered_preds[self.p_labels == int(pred_i)] = int(target_i)
        s_correct_idx = (reordered_preds[labeled_idx] == self.targets[labeled_idx])  # 伪标签无变化；无标签部分
        w_correct_idx = (reordered_preds[unlabeled_idx] == self.targets[unlabeled_idx])
        self.s_acc = s_correct_idx.mean()
        self.w_acc = w_correct_idx.mean()
        print("Strong pseudo Label Accuracy {:.2f}".format(100 * self.s_acc) + "%")
        print("Weak pseudo Label Accuracy {:.2f}".format(100 * self.w_acc) + "%")


        # Rink
        probs_iter = F.normalize(torch.tensor(Z), 1).numpy()  # 将某一个维度除以那个维度对应的2范数, 按行操作

        # 直接按照类别，对每一个 P/P范数 的特征预测进行排序
        # P_l2 = np.linalg.norm(probs_iter, ord=2, axis=0)
        P_s, idx = torch.sort(torch.from_numpy(probs_iter), dim=0, descending=True)  # 按照第一维(每个预测类别内)进行排序
        idx_groupi_n = []
        for i in range(len(self.dataset.classes)):
            num_labels = int(num * len(P_s[:, i]))
            idx_groupi_n.append(idx[0:num_labels, i])  # 内部取num
        P_labeled_idx = torch.cat(idx_groupi_n, dim=0)  # 连接成完整的new_labeled_idx

        if epoch == 0:
            _labeled_idx = np.setdiff1d(labeled_idx, P_labeled_idx) # labeled_idx去掉相交的部分
            new_labeled_idx = np.setdiff1d(labeled_idx, _labeled_idx) # 拿到与labeled_idx相交的部分
            labeled_idx = new_labeled_idx
            self.labeled_idx = labeled_idx # 更新
            # print(self.labeled_idx)
            self.unlabeled_idx = np.setdiff1d(np.arange(len(self.dataset)), self.labeled_idx) # 更新有标签和无标签部分
            unlabeled_idx = self.unlabeled_idx


        # 只需要找到对应的新加入的位置的值就好了, 然后argmax变成标签。可能是0.98这种, 可能需要label平滑
        ### Extract and test pseudo labels for accuracy
        # probs_iter = F.normalize(torch.tensor(Z), 1).numpy()# 将某一个维度除以那个维度对应的2范数, 按行操作
        probs_iter[labeled_idx] = np.zeros(len(self.dataset.classes)) # ！!! 初始化为零,probs_iter顺序无变化
        probs_iter[labeled_idx, labels[labeled_idx]] = 1 # ！！！将标签设置为1，重置
        self.probs_iter = probs_iter # ！！！
        # # 初始化有标签的标签
        p_labels = np.argmax(probs_iter, 1) #argmax函数 # 按行方向搜索最大值, 并返回索引;顺序无变化
        self.p_labels = p_labels # ！！！ 得到第 j 位置为 1 的标签, 真正的伪标签


        ### Calculate Pseudo Label Accuracy
        from utils.evaluate_utils import _hungarian_match
        match = _hungarian_match(self.p_labels, self.targets, preds_k=len(self.dataset.classes),
                                 targets_k=len(self.dataset.classes))
        reordered_preds = np.zeros_like(self.p_labels)
        for pred_i, target_i in match:
            reordered_preds[self.p_labels == int(pred_i)] = int(target_i)
        # correct_idx = (reordered_preds[unlabeled_idx] == self.targets[unlabeled_idx]) #伪标签无变化；无标签部分
        # self.acc = correct_idx.mean()
        # print("New Pseudo Label Accuracy {:.2f}".format(100 * self.acc) + "%")
        s_correct_idx = (reordered_preds[labeled_idx] == self.targets[labeled_idx])  # 伪标签无变化；无标签部分
        w_correct_idx = (reordered_preds[unlabeled_idx] == self.targets[unlabeled_idx])
        self.s_acc = s_correct_idx.mean()
        self.w_acc = w_correct_idx.mean()
        print("New Strong pseudo Label Accuracy {:.2f}".format(100 * self.s_acc) + "%") # 没什么变化，很奇怪啊
        print("New Weak pseudo Label Accuracy {:.2f}".format(100 * self.w_acc) + "%")

        # after propagating, get acc of the points of different density, 0.8-0.99
        rp_label_acc = (reordered_preds_all[labeled_idx] == self.targets[labeled_idx]).mean()
        low_acc = (reordered_preds_all[low_index] == self.targets[low_index]).mean()
        if len(high_index) > 0:
            high_acc = (reordered_preds_all[high_index] == self.targets[high_index]).mean()
        else:
            high_acc = 0
        middle_acc = (reordered_preds_all[middle_index] == self.targets[middle_index]).mean()
        print('after propagating, rp_label_acc:{:.2f}, low_acc:{:.2f}, middle_acc:{:.2f}, high_acc:{:.2f}'.format(
            rp_label_acc, low_acc, middle_acc, high_acc))
        print('after propagating, rp_num:{}, low_num:{}, middle_num:{}, high_num:{}'.format(len(labeled_idx),
                                                                                            len(low_index),
                                                                                            len(middle_index),
                                                                                            len(high_index)))

        # Compute the knn accuracy of low_index, middle_index, high_index
        targets = self.targets

        # Compute the knn accuracy of all samples
        neighbor_targets = np.take(targets, I, axis=0)  # Exclude sample itself for eval
        anchor_targets = np.repeat(targets.reshape(-1, 1), k, axis=1)
        accuracy = np.mean(neighbor_targets == anchor_targets)
        print("Knn acc of all samples:{:.2f}".format(accuracy))

        neighbor_targets = np.take(targets, I[labeled_idx], axis=0)  # Exclude sample itself for eval
        anchor_targets = np.repeat(targets[labeled_idx].reshape(-1, 1), k, axis=1)
        acc_rp_label = np.mean(neighbor_targets == anchor_targets)
        print("Knn acc and number of rp_labeled index:{:.2f}, {}".format(acc_rp_label, len(labeled_idx)))

        neighbor_targets = np.take(targets, I[low_index], axis=0)  # Exclude sample itself for eval
        anchor_targets = np.repeat(targets[low_index].reshape(-1, 1), k, axis=1)
        acc_low = np.mean(neighbor_targets == anchor_targets)
        print("Knn acc and number of low index:{:.2f}, {}".format(acc_low, len(low_index)))

        neighbor_targets = np.take(targets, I[middle_index], axis=0)  # Exclude sample itself for eval
        anchor_targets = np.repeat(targets[middle_index].reshape(-1, 1), k, axis=1)
        acc_middle = np.mean(neighbor_targets == anchor_targets)
        print("Knn acc and number of middle index:{:.2f}, {}".format(acc_middle, len(middle_index)))

        acc_high = 0
        if len(high_index) > 0:
            neighbor_targets = np.take(targets, I[high_index], axis=0)  # Exclude sample itself for eval
            anchor_targets = np.repeat(targets[high_index].reshape(-1, 1), k, axis=1)
            acc_high = np.mean(neighbor_targets == anchor_targets)
            print("Knn acc and number of high index:{:.2f}, {}".format(acc_high, len(high_index)))


    def dis_align(self, probs_iter):
        probs_iter = F.normalize(torch.tensor(probs_iter), 1).numpy()
        p_labels = np.argmax(probs_iter, axis=1)
        labeled_idx = np.asarray(self.labeled_idx)
        unlabeled_idx = np.asarray(self.unlabeled_idx)
        labels = np.asarray(self.all_labels)

        label_dis_l = np.zeros(len(self.dataset.classes))
        for i in labeled_idx:
            label_dis_l[labels[i]] += 1
        label_dis_l = label_dis_l / len(labeled_idx)

        for i in range(100):
            label_dis_u = np.zeros(len(self.dataset.classes))
            for i in unlabeled_idx:
                label_dis_u[p_labels[i]] += 1
            label_dis_u = label_dis_u / len(unlabeled_idx)

            label_dis = np.divide(label_dis_l, label_dis_u + 0.0000001)
            label_dis[label_dis > 1.01] = 1.01
            label_dis[label_dis < 0.99] = 0.99

            for i in range(len(self.dataset.classes)):
                probs_iter[unlabeled_idx, i] = probs_iter[unlabeled_idx, i] * label_dis[i]

            probs_iter = F.normalize(torch.tensor(probs_iter), 1).numpy()
            p_labels = np.argmax(probs_iter, axis=1)
        return probs_iter



def create_data_loaders_simple(args, p, train_dataset, labels, l_indices, indices, data=None, checkpoint=False):
    if checkpoint:
        dataset = data
    else:
        dataset = DBSS(args, train_dataset, labels, l_indices)

    sampler = SubsetRandomSampler(dataset.labeled_idx)
    batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
    train_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=p['num_workers'], pin_memory=True) # 只有有标签的部分
    train_loader_noshuff = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,num_workers=p['num_workers'], pin_memory=True,drop_last=False)

    batch_sampler_l = StreamBatchSampler(dataset.labeled_idx, batch_size=args.labeled_batch_size)
    batch_sampler_u = BatchSampler(SubsetRandomSampler(dataset.unlabeled_idx), batch_size=args.batch_size-args.labeled_batch_size, drop_last=True)

    train_loader_l = DataLoader(dataset, batch_sampler=batch_sampler_l, num_workers=p['num_workers'], pin_memory=True)
    train_loader_u = DataLoader(dataset, batch_sampler=batch_sampler_u, num_workers=p['num_workers'], pin_memory=True)


    return train_loader, train_loader_noshuff, train_loader_l,train_loader_u, dataset




def mixup_data(x_1, index, lam):
    mixed_x_1 = lam * x_1 + (1 - lam) * x_1[index, :]
    return mixed_x_1

def mixup_criterion(pred, y_a, y_b, lam):
    criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def train_semi(p, train_loader_l, train_loader_u, model, optimizer, epoch, global_step, args, ema_model=None):
    model.train()
    lr_length = len(train_loader_u) # train_loader_u改train_loader_l
    train_loader_l = iter(train_loader_l)

    if args.progress == True:
        tk0 = tqdm(train_loader_u, desc="Semi Supervised Learning Epoch " + str(epoch) + "/" + str(args.epochs),
                   unit="batch", ncols=80)
        loss_meter = meter.AverageValueMeter()
    else:
        tk0 = train_loader_u

    # Training
    for i, (aug_images_u, target_u) in enumerate(tk0):
        aug_images_l, target_l = next(train_loader_l) #self.plables
        target_l = target_l.to(args.device)
        target_u = target_u.to(args.device)
        target = torch.cat((target_l, target_u), 0)
        alpha = args.alpha
        index = torch.randperm(args.batch_size, device=args.device)
        lam = np.random.beta(alpha, alpha)
        target_a, target_b = target, target[index]
        optimizer.zero_grad()
        adjust_learning_rate(optimizer, epoch, i, lr_length, args)
        count = 0
        for batch_l, batch_u in zip(aug_images_l, aug_images_u):
            batch_l = batch_l.to(args.device) # (32,3,96,96)
            batch_u = batch_u.to(args.device) # (16,3,96,96)
            batch = torch.cat((batch_l, batch_u), 0)
            m_batch = mixup_data(batch, index, lam)
            class_logit = model(m_batch)
            class_logit = class_logit[0]

            if count == 0:
                loss_sum = mixup_criterion(class_logit.double(), target_a, target_b, lam).mean()
            else:
                loss_sum = loss_sum + mixup_criterion(class_logit.double(), target_a, target_b, lam).mean()
            count += 1

        loss = loss_sum / (args.aug_num)
        loss.backward()# retain_graph = True
        optimizer.step()
        if args.progress == True:
            loss_meter.add(loss.item())
            tk0.set_postfix(loss=loss_meter.mean)
        global_step += 1
    print('loss:\t{:.4f}'.format(loss_meter.mean))
    return global_step, loss_meter.mean


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def adjust_learning_rate(optimizer, epoch, step_in_epoch, total_steps_in_epoch, args):
    lr = args.lr
    epoch = epoch + step_in_epoch / total_steps_in_epoch

    # Cosine LR rampdown from https://arxiv.org/abs/1608.03983 (but one cycle only)
    if args.lr_rampdown_epochs:
        assert args.lr_rampdown_epochs >= args.epochs
        lr *= cosine_rampdown(epoch, args.lr_rampdown_epochs)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# 通用的获取特征的函数
def extract_features_simp(train_loader, model, args, forward_pass='backbone'):
    model.eval()
    embeddings_all = []
    target_all = []

    with torch.no_grad():
        for i, (batch_input) in enumerate(train_loader):
            X_n = batch_input['image'].cuda()
            Y_n = batch_input['target']
            if forward_pass == 'default':
                feats = model(X_n)
                feats = feats[0]

            elif forward_pass == 'backbone':
                feats = model(X_n, forward_pass=forward_pass)
            embeddings_all.append(feats.data.cpu())
            target_all.append(Y_n)
    embeddings_all = np.asarray(torch.cat(embeddings_all).numpy())
    target_all = np.asarray(torch.cat(target_all).numpy())
    
    if forward_pass == 'backbone':
        return embeddings_all, target_all
    return embeddings_all