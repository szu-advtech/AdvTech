"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import scipy
import torch
import numpy as np
import errno
import torch.nn as nn
import sys
import torch.nn.functional as F
from faiss import normalize_L2
import faiss

class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        # fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)
        output = model(images)
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))


def confusion_matrix(predictions, gt, class_names, output_file=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1)
    
    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix, cmap='Blues')
    axes.set_xticks([i for i in range(len(class_names))])
    axes.set_yticks([i for i in range(len(class_names))])
    axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
    axes.set_yticklabels(class_names, ha='right', fontsize=8)
    
    for (i, j), z in np.ndenumerate(confusion_matrix):
        if i == j:
            axes.text(j, i, '%d' %(100*z), ha='center', va='center', color='white', fontsize=6)
        else:
            pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


# based on centers
# def get_distant_neighbors(batch: torch.Tensor, anchors_features: torch.Tensor, centers, dk):
#
#     sim_cos = nn.CosineSimilarity(dim=2, eps=1e-6)  # cosine similarity
#     similarity = sim_cos(anchors_features.unsqueeze(1), centers.unsqueeze(0))
#     sorted, indices = torch.sort(similarity, dim=1, descending=True)
#     ind_max = indices[:, 0]
#     anchor_center = centers[ind_max]   # each anchor's center
#
#     similarity2 = sim_cos(anchor_center.unsqueeze(1), anchors_features.unsqueeze(0))  # the cosine similarity between each anchor_center and anchors_features
#
#     _, indices2 = torch.sort(similarity2, dim=1, descending=True)
#     # sub = 5
#     # inds = np.random.choice(-1*np.arange(1,dk+1), sub)
#     # inds = torch.from_numpy(inds)
#     # ind_min = indices2[:,inds]
#     ind_min = indices2[:, -10:]
#     return ind_min


# get the distant neighbors based on pseudo labels
def get_distant_neighbors(batch: torch.Tensor, anchors_output: torch.Tensor, epoch, labels):

    sim_cos = nn.CosineSimilarity(dim=2, eps=1e-6)  # cosine similarity
    similarity = sim_cos(anchors_output.unsqueeze(1), anchors_output.unsqueeze(0))
    sorted, indices = torch.sort(similarity, dim=1, descending=True)

    dk = min(1+int(epoch/25), 4)
    ind_labels = labels[indices]
    anchors_output = anchors_output[indices]

    tag = np.arange(0,10)
    disks_output = []
    for i in range(ind_labels.shape[0]):
        tag_ = list(set(tag) - set([ind_labels[i][0].item()]))
        mask = ind_labels[i].view(1,-1).repeat(len(tag_), 1) == torch.tensor(tag_).view(-1,1).cuda()
        disks_output_i = [anchors_output[i][mask[j]][-dk:] for j in range(len(mask))]
        disks_output_i = torch.cat(disks_output_i, dim=0)
        # when some clusters's samples are not enough
        if disks_output_i.shape[0] < len(tag_)*dk:
            num = len(tag_)*dk - disks_output_i.shape[0]
            disks_output_i = torch.cat([disks_output_i, disks_output_i[-num:]], dim=0)
        disks_output.append(disks_output_i.unsqueeze(0))
    disks_output = torch.cat(disks_output, dim=0)

    return disks_output

# get the centers of batch based on the pseudo labels
def get_batch_centers(anchors_output: torch.Tensor, pseudo_labels):
    values, _ = torch.max(anchors_output, dim=1)
    sorted, indices = torch.sort(values, dim=0, descending=True)
    labels = pseudo_labels[indices]
    anchors_output_sorted = anchors_output[indices]
    classes = torch.unique(labels)
    centers = []
    centers_ratio = 0.5
    for i in range(classes.shape[0]):
        mask = labels == classes[i]
        num = int(torch.sum(mask) * centers_ratio)
        centers.append(anchors_output_sorted[mask][0:num,:].mean(axis=0).unsqueeze(dim=0))
    centers = torch.cat(centers, dim=0)
    return centers


@torch.no_grad()
def count_negative_simples(disks_index, targets):
    if len(disks_index.shape) == 2:
        neg_num = disks_index.shape[0] * disks_index.shape[1]
    else:
        neg_num = disks_index.shape[0]

    true_neg_num = torch.tensor([0], dtype=torch.int64).cuda()
    for i in range(disks_index.shape[0]):
        true_neg_num += torch.sum(targets[i] != targets[disks_index[i]])

    return true_neg_num.item(), neg_num

@torch.no_grad()
def get_features_train(train_loader, model, forward_pass='default'):
    model.train()
    targets, features, indices = [], [], []
    for i, batch in enumerate(train_loader):
        # Forward pass
        input_ = batch['image'].cuda()
        target_ = batch['target'].cuda()
        index_ = batch['index'].cuda()
        with torch.no_grad():
            feature_ = model(input_, forward_pass = forward_pass)

        if forward_pass == 'default':
            feature_ = feature_[0]

        targets.append(target_)
        features.append(feature_)
        indices.append(index_)

    targets = torch.cat(targets).int()
    features = torch.cat(features)
    indices = torch.cat(indices)

    features_order = torch.zeros_like(features)
    features_order[indices] = features

    targets_order = torch.zeros_like(targets)
    targets_order[indices] = targets

    return features_order, targets_order

@torch.no_grad()
def get_features_eval(val_loader, model, forward_pass='default'):
    model.eval()
    targets, features, indices = [], [], []
    for i, batch in enumerate(val_loader):
        # Forward pass
        input_ = batch['image'].cuda()
        target_ = batch['target'].cuda()
        # index_ = batch['index'].cuda()
        index_ = batch['meta']['index']

        with torch.no_grad():
            feature_ = model(input_, forward_pass = forward_pass)

        if forward_pass == 'default':
            feature_ = feature_[0]

        targets.append(target_)
        features.append(feature_.cpu())
        indices.append(index_)

    targets = torch.cat(targets).int()
    features = torch.cat(features)
    indices = torch.cat(indices)

    features_order = torch.zeros_like(features)
    features_order[indices] = features

    targets_order = torch.zeros_like(targets)
    targets_order[indices] = targets

    return features_order, targets_order

@torch.no_grad()
def select_samples(feas_sim, scores, ratio_select, num_cluster=10, center_ratio=0.5):
    _, idx_max = torch.sort(scores, dim=0, descending=True)
    idx_max = idx_max.cpu()
    num_per_cluster = idx_max.shape[0] // num_cluster
    k = int(num_per_cluster * center_ratio)
    idx_max = idx_max[0:k, :]

    centers = []
    for c in range(num_cluster):
        centers.append(feas_sim[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

    centers = torch.cat(centers, dim=0)

    num_select_c = int(num_per_cluster * ratio_select)

    dis = torch.einsum('cd,nd->cn', [centers, feas_sim])
    idx_select = torch.argsort(dis, dim=1, descending=True)[:, 0:num_select_c].flatten()
    labels_select = torch.arange(0, num_cluster).unsqueeze(dim=1).repeat(1, num_select_c).flatten()

    return idx_select, labels_select


@torch.no_grad()
def select_centers(features, scores, num_cluster=10, center_ratio=0.4):
    _, idx_max = torch.sort(scores, dim=0, descending=True)
    idx_max = idx_max.cpu()
    num_per_cluster = idx_max.shape[0] // num_cluster
    k = int(num_per_cluster * center_ratio)
    idx_max = idx_max[0:k, :]

    centers = []
    for c in range(num_cluster):
        centers.append(features[idx_max[:, c], :].mean(axis=0).unsqueeze(dim=0))

    centers = torch.cat(centers, dim=0)

    return centers


@torch.no_grad()
def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    # 得到一个bbox和原图的比例
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 得到bbox的中心点
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    # 代码实现中有一些不同的是，生成bbox的中心点是在全图范围随机，如果中心点靠近图像边缘，那么bbox的面积和原图的比可能就不是1−λ。因此这个面积比例是重新计算的。
    #
    # 图像之间的对应关系是随机的，有可能对应到自己本身，就不会进行cutmix，多执行几次能看到效果。
    return bbx1, bby1, bbx2, bby2


@torch.no_grad()
def get_knn_indices(base_dataloader, model, topk=20):
    from utils.utils import get_features_eval
    features, targets = get_features_eval(base_dataloader, model, forward_pass='backbone')

    from utils.evaluate_utils import kmeans
    kmeans(features, targets)

    import faiss
    features = features.cpu().numpy()
    n, dim = features.shape[0], features.shape[1]
    index = faiss.IndexFlatIP(dim)  # index = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_all_gpus(index)
    index.add(features)
    distances, indices = index.search(features, topk + 1)  # Sample itself is included

    # evaluate
    targets = targets.cpu().numpy()
    neighbor_targets = np.take(targets, indices[:, 1:], axis=0)  # Exclude sample itself for eval
    anchor_targets = np.repeat(targets.reshape(-1, 1), topk, axis=1)
    accuracy = np.mean(neighbor_targets == anchor_targets)

    return indices, accuracy


@torch.no_grad()
def get_pseudo_labels(predictions, match):
    predictions = predictions[0]
    probs = predictions['probabilities'] # cpu ，每一行总和为1，[[0.5,0.5]..]
    max_prob, _ = torch.max(probs, dim=1)# cpu, 取每一行概率最大的
    plabels = predictions['predictions'] # cpu
    targets = predictions['targets'] # cuda0
    pl_indices = predictions['indices'] # 只包含符合条件的伪标签(如、710713=99.58),cuda0


    # evaluate the accuracy of the pseudo-labels
    reordered_preds = torch.zeros(len(plabels), dtype=plabels.dtype)
    for pred_i, target_i in match:
        reordered_preds[plabels == int(pred_i)] = int(target_i)
    acc = sum(reordered_preds == targets)/len(reordered_preds)*100 # reordered_preds:cpu, targets:cpu
    print("accuracy of the pseudo-labels:{}/{}\t{:.2f}".format(sum(reordered_preds == targets), len(reordered_preds), acc))

    # get equal number of pseudo label for every class and the number equals to the min of number of all classes.
    num_classes = torch.unique(targets).numel()# .numel() = len() 取tensor长度
    num = min([sum(plabels == i).item() for i in range(num_classes)]) # print number,item()将转化为python数字，这里将会获得伪标签正确的数量的最小值
    Snumber = 0
    for i in range(num_classes):
        Snumber += sum(plabels == i).item() # 各个类的伪标签数量
        print("{}类别的伪标签数量： {}".format(i, sum(plabels == i).item()))
    print("所有类别的伪标签数量： {}".format(Snumber))
    print("所有类别的伪标签min数量： {}".format(num))

    max_prob_group = [max_prob[plabels == i] for i in range(num_classes)] # 获得max_prob中伪标签对应 真 是标签的值且概率最大的类别的值
    pl_indices_group = [pl_indices[plabels == i] for i in range(num_classes)] # 按类别将伪标签和对应的索引划分集合,pl_indices_group有顺序。
    plabels_n = []
    pl_indices_n = []
    for i in range(num_classes): # (排序，取前topk个)将所有类别的伪标签数量写成一样多，比如标签为1的样本数量和样本3的样本数量一致，其余也一样。
        val, ind = torch.topk(max_prob_group[i], num) # 取一个tensor的topk元素（降序后的前k个大小的元素值及索引,由大到小
        # import random
        # ind = random.sample(list(range(0,len(max_prob_group[i]))), num)
        plabels_n.append(torch.ones(num).to(torch.int64) * i) # [[1,1,1][2,2,2][3,3,3]] 按照顺序将伪标签放好。
        pl_indices_n.append(pl_indices_group[i][ind]) # pl_indices_group本来就有顺序，只是需要冲对应类别中对应的索引。

    plabels = torch.cat(plabels_n) # 变成一维张量
    pl_indices = torch.cat(pl_indices_n)

    return plabels, pl_indices


def L2_dist(X, Y):
    shape = X.shape
    xx = np.sum(np.multiply(X, X), 1).reshape([shape[0], 1])
    yy = np.sum(np.multiply(Y, Y), 1).reshape([shape[0], 1])
    xy = np.matmul(X, Y.T)
    D = xx + yy.T - 2 * xy
    D[D < 0] = 0
    return D


# 使用faiss函数计算得其近邻样本集合
def get_knn(k, X, l2=False, index_type="ip", n_labels=None):
    if l2:
        normalize_L2(X)

    # kNN search for the graph
    d = X.shape[1]
    if index_type == "ip":
        index = faiss.IndexFlatIP(d)  # build the index
    if index_type == "l2":
        index = faiss.IndexFlatL2(d)  # build the index
    ngus = faiss.get_num_gpus()
    index = faiss.index_cpu_to_all_gpus(index, ngpu=ngus)
    index.add(X) # 通过feature来找近邻
    D, I = index.search(X, k + 1)  # 计算得其k个近邻样本集合

    return D, I


def cal_knn_acc(W, targets, label_index, low_index, middle_index, high_index):
    all_sum, label_sum, low_sum, middle_sum, high_sum = 0, 0, 0, 0, 0
    all_right, label_right, low_right, middle_right, high_right = 0, 0, 0, 0, 0

    for i in label_index:
        label_pred = targets[np.where(W[i]>0)]
        label_right += (sum( label_pred == targets[i])-1)  # minus itself
        label_sum += (len(label_pred) -1)

    for i in low_index:
        low_pred = targets[np.where(W[i]>0)]
        low_right += (sum( low_pred == targets[i])-1)  # minus itself
        low_sum += (len(low_pred) -1)

    for i in middle_index:
        middle_pred = targets[np.where(W[i]>0)]
        middle_right += (sum( middle_pred == targets[i])-1)  # minus itself
        middle_sum += (len(middle_pred) -1)

    for i in range(len(targets)):
        all_pred = targets[np.where(W[i]>0)]
        all_right += (sum( all_pred == targets[i])-1)
        all_sum += (len(all_pred)-1)

    label_acc = np.round(label_right/label_sum, 2)
    low_acc = np.round(low_right / low_sum, 2)
    middle_acc = np.round(middle_right / middle_sum, 2)
    u_all_acc = np.round((low_right+middle_right+high_right)/(low_sum+middle_sum+high_sum), 2)
    all_acc = np.round(all_right/all_sum, 2)
    print('all_acc:{}, label_acc:{}, u_all_acc:{}, low_acc:{}, middle_acc:{}'.format(all_acc, label_acc, u_all_acc, low_acc, middle_acc))


def assign_true_knn_indices(D, I, targets, k):
    N = len(targets)
    D1, I1 = torch.from_numpy(D), torch.from_numpy(I)
    t0 = targets[I1[:,0].view(-1,1).expand_as(I1)]
    tn = targets[I1]
    mask = t0==tn
    D2 = torch.zeros(N,k+1)
    I2 = torch.zeros(N,k+1)
    for i in range(N):
        if len(D1[i][mask[i]]) >= k + 1:
            D2[i] = D1[i][mask[i]][:k+1]
            I2[i] = I1[i][mask[i]][:k+1]
        else:
            D2[i] = D1[i][:k+1]
            I2[i] = I1[i][:k+1]
    D1, I1 = D2.numpy(), I2.numpy().astype(np.int64)

    return D1, I1


def update_W_AND(H, W0, gamma=0):
    H_T = np.transpose(H)
    S = np.matmul(H, H_T)
    mu = np.mean( S[W0 > 0] )
    std = np.std( S[W0 > 0], ddof=1 )
    print('mu:{:.2f},std:{:.2f}'.format(mu,std))
    threshold = mu + gamma * std
    # S[S < threshold] = 0
    # W0 = S
    W0[S>=threshold] = 1
    percent = (W0>0).sum() / (W0.shape[0] * W0.shape[1])
    return W0, percent, threshold