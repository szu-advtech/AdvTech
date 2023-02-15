"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import copy
import numpy as np
import scipy
import torch
import torch.nn.functional as F
from utils.common_config import get_feature_dimensions_backbone
from utils.utils import AverageMeter, confusion_matrix
from data.custom_dataset import NeighborsDataset
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from losses.losses import entropy
from sklearn.cluster import KMeans


@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        images = batch['image'].cuda(non_blocking=True)
        target = batch['target'].cuda(non_blocking=True)

        output = model(images)
        output = memory_bank.weighted_knn(output) 

        acc1 = 100*torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), images.size(0))

    return top1.avg


@torch.no_grad()
def get_predictions(p, dataloader, model, forward_type='online_evaluation', return_features=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    targets = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()

    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
        key_ = 'anchor' # 直接无关是否有那个neighbor
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'image'
        include_neighbors = False

    ptr = 0
    for batch in dataloader:
        images = batch[key_].cuda(non_blocking=True)
        bs = images.shape[0]
        res = model(images, forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr: ptr + bs] = res['features']
            ptr += bs
        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))
        targets.append(batch['target'])
        if include_neighbors:
            neighbors.append(batch['anchor_neighbors_indices'])
    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)

    if include_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': neighbors} for
               pred_, prob_ in zip(predictions, probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets} for pred_, prob_ in
               zip(predictions, probs)]

    if return_features:
        return out, features.cpu()
    else:
        return out


@torch.no_grad()
def get_predictions_propa(args, p, dataloader, model, return_features=True):
    model.train()
   
    targets = []
    indices = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()

    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'image'
        include_neighbors = False
    
    ptr = 0
    sum = 0
    for batch in dataloader:
        images = batch[key_].cuda(non_blocking=True)
        bs = images.shape[0] # bs = 200 = batch_size
        res = model(images, forward_pass='return_all')
        output = res['output'] # 返回“output”部分的值


        if return_features: # 复制，同时也可以转化为torch
            features[ptr: ptr + bs] = res['features'] # features(200,512)
            ptr += bs

        targets.append(batch['target'].cuda()) # batch['target']是cpu且后面不可以更换这个设备分布
        indices.append(batch['index'].cuda()) # 根据掩码张量mask中的二元值，取输入张量中的指定项( mask为一个 ByteTensor)，将取值返回到一个新的1D张量，张量 mask须跟input张量有相同数量的元素数目，但形状或维度不需要相同

    device = torch.device("cpu")
    targets = torch.cat(targets, dim = 0) # 此时是gpu的tensor
    indices = torch.cat(indices, dim = 0)
    targets = targets.to(device) # 必须要赋值才会有效果
    indices = indices.to(device)


    # 将获得的特征构造G=(V,E),E由l2范数获得    ?这里需要迭代吗，还是只有一次，还是最好多来几次吧
    #     获得W,S,C
    #     进行标签传播
    #     返回那个最后概率转移矩阵P
    # 排序，根据P-Y
    # 取前10%的标签作为强伪标签，将标签plabels和索引pl_indices拿出
    # # 构造强弱伪标签数据集

    
    num_sample = len(features)
    alpha = 0.99
    max_iter = 40;
    # graph construction
    W = buildGraph(features, args.sigma)
    # Normalize the graph
    W = W - scipy.sparse.diags(W.diagonal())
    S = W.sum(axsi=1)
    S[S==0]  = 1
    D = np.array(1. / np.sqrt(S)) # 度矩阵
    D = scipy.sparse.diags(D.reshape(-1))
    Wn = D * W * D

    # Initiliaze the y vector for each class
    labels = targets
    Z = np.zeros((num_sample, p['num_classes']))
    A = scipy.sparse.eye(Wn.shape[0]) - alpha * Wn # # 返回一个稀疏 (m x n) 矩阵，其中第 k 个对角线全为 1，其他全为 0。
    Y = np.zeros((num_sample, p['num_classes']))
    for i in range(num_sample):
        Y[i][labels[i]] = 1.0
    
    # 共轭梯度求解
    for i in range(p['num_classes']):
        f, _ = scipy.sparse.linalg.cg(A, Y[:, i], tol=1e-3, max_iter=max_iter)
        Z[:, i] = F
    Z[Z < 0] = 0

    # result
    probs_iter = F.normalize(torch.tensor(Z), 1).numpy()
    P, p_labels = torch.max(probs_iter, dim =1)
    
    # 计算P-Y
    P_Y = P - Y # tensor可以直接相加减
    # rank
    P_Ys = torch.sort(P_Y, 1) # 每行按照升序排列
    P_Ys = P_Ys[:,0] # 取最前面一列
    P_Yv, P_Yi = torch.sort(P_Ys, 1, descending=True) # 对一列进行逆序排序，排前面的越大，说明越靠近1，越真实
    p_labels_s = p_labels[P_Yi] # 按照P_Yi的值进行取值，即按指定索引排序

    # extract
    num_labels = 0.1 * num_sample
    labels_idx = p_labels_s[0:num_labels, :]
    pplables = P_Yv[0:num_labels,:]
    # unlabels_idx = np.setdiff1d(np.arange(num_sample), labels_idx)
    return pplables, labels_idx

    
    # 需要所有的标签，但是有一个问题，就是标签传播需要没有标签的值，但是可以这么理解，有了标签之后再标签传播，会得到的是基于近邻的标签。
    # 这里的标签从哪里来，这里无监督学习是没有标签的说法的，所以只能是前一阶段预测得到的伪标签（不分强弱，用于标签传播）。
    



    
    # clamp_data_label = np.zeros((num_sample, p['num_classes']), np.float32)
    # for i in range(num_sample):
    #     clamp_data_label[i][targets[i]] = 1.0

    # label_function = np.zeros((num_sample, p['num_classes']), np.float32)
    # label_function = []
    # label_function = copy.copy(np.asarray(clamp_data_label))

    # P = np.zeros(len(feats), len(feats), np.float32)
    # changed = np.abs(P - ).sum()
    # while iter < max_iter and changed > tol:
    #     if iter % 1==0:
    #         print("---> Iteration %d/%d, changed: %f" )% (iter, max_iter, changed)
    #     P = label_function
    #     iter += 1
    #     # propaganda
    #     label_function  = np.dot(W, label_function)
    #     # clamp 重置
    #     label_function[0: num] = clamp_data_label # Y
    #     # check converge
    #     changed = np.abs(P - label_function).sum()
    
    # plabels_data = np.zeros()# 无标签的数量
    # for i in range(len(feats)):
    #     plabels_data[i] = np.argmax(label_function[i])

    # if include_neighbors:
    #     neighbors = torch.cat(neighbors, dim=0)
    #     out = [{'targets': targets, 'indices': indices}]
    #     print(1)

    # else:
    #     out = [{'targets': targets, 'indices': indices} ]

    # if return_features:
    #     return out, features.cpu()
    # else:
    #     return out
     

@torch.no_grad()
def  get_predictions_scan_example(p, dataloader, model, ct, forward_type='online_evaluation',  return_features=False):
    # Make predictions on a dataset with neighbors
    model.train()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    targets = []
    indices = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()

    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'image'
        include_neighbors = False

    ptr = 0
    sum = 0
    for batch in dataloader:
        images = batch[key_].cuda(non_blocking=True)
        bs = images.shape[0] # bs = 200 = batch_size
        res = model(images, forward_pass='return_all')
        output = res['output'] # 返回“output”部分的值


        if return_features: # 复制，同时也可以转化为torch
            features[ptr: ptr + bs] = res['features'] # features(200,513)
            ptr += bs

        mask = None
        for i, output_i in enumerate(output): # output_i(200,10),output_i为模型各个类别的预测概率
            max_prob, target = torch.max(output_i.softmax(1), dim=1) # 第一维度返回一行最大值prob以及对应的（10索引即标签）标签。
            mask = max_prob > ct # 如果大于设置的阈值，则说明是强伪标签
            predictions[i].append(torch.masked_select(torch.argmax(output_i, dim=1), mask.squeeze())) # 会发现,输入的argmax(output_i)会根据 mask 对应位置的值 进行删选；这里返回符合的坐标
            probs[i].append(F.softmax(output_i, dim=1)[mask]) # 对每一行进行softmax --- dim = 1轴，每一行总和为1
            sum += torch.sum(mask > 0)

        # # 对第一阶段获得的聚类指示概率进行阈值截断
        # for i, output_i in enumerate(output): # output为第一阶段模型最后一个线性层（dim->10）结果
        #     max_prob, target = torch.max(output_i.softmax(1), dim=1) # 第一维度返回一行最大值prob以及对应的（10索引即标签）标签。
        #     mask = max_prob > ct # 构建掩码：如果大于设置的阈值，则暂时加入强伪标签数据集，mask对应位置设置为1
        #     predictions[i].append(torch.masked_select(torch.argmax(output_i, dim=1), mask.squeeze())) # 按照掩码取出结果
        #     probs[i].append(F.softmax(output_i, dim=1)[mask])
        #     sum += torch.sum(mask > 0)


        targets.append(torch.masked_select(batch['target'].cuda(), mask.squeeze())) # batch['target']是cpu且后面不可以更换这个设备分布
        indices.append(torch.masked_select(batch['index'].cuda(), mask.squeeze())) # 根据掩码张量mask中的二元值，取输入张量中的指定项( mask为一个 ByteTensor)，将取值返回到一个新的1D张量，张量 mask须跟input张量有相同数量的元素数目，但形状或维度不需要相同


        if include_neighbors:
            batch1 = batch['anchor_neighbors_indices'].cuda()  # 2022/11/19
            neighbors.append(batch1[mask])
            # neighbors.append(batch['anchor_neighbors_indices'][mask] )

    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    # targets = [torch.cat(target_, dim=0).cpu() for target_ in targets]
    # indices = [torch.cat(indice_, dim=0).cpu() for indice_ in indices]
    device = torch.device("cpu")
    targets = torch.cat(targets, dim = 0) # 此时是gpu的tensor
    indices = torch.cat(indices, dim = 0)
    targets = targets.to(device) # 必须要赋值才会有效果
    indices = indices.to(device)
    # targets.cpu() #转成cpu的tensor
    # indices.cpu()


    if include_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': neighbors, 'indices': indices} for
               pred_, prob_ in zip(predictions, probs)]
        print(1)

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'indices': indices} for pred_, prob_ in
               zip(predictions, probs)]

    if return_features:
        return out, features.cpu()
    else:
        return out, sum

@torch.no_grad()
def  get_predictions_our_example(p, dataloader, model, ct, forward_type='online_evaluation',  return_features=False):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    targets = []
    indices = []
    alphas = []
    if return_features:
        ft_dim = get_feature_dimensions_backbone(p)
        features = torch.zeros((len(dataloader.sampler), ft_dim)).cuda()

    if isinstance(dataloader.dataset, NeighborsDataset):  # Also return the neighbors
        key_ = 'anchor'
        include_neighbors = True
        neighbors = []

    else:
        key_ = 'image'
        include_neighbors = False

    from utils.utils import get_features_eval
    neighbors_outputs, _ = get_features_eval(dataloader, model)

    ptr = 0
    sum = 0
    for batch in dataloader:
        images = batch[key_].cuda(non_blocking=True) # cuda(0)

        bs = images.shape[0]
        res = model(images, forward_pass='return_all')
        output = res['output']
        if return_features:
            features[ptr: ptr + bs] = res['features']
            ptr += bs

        ct1 = 0.9
        eta = 0.6
        neighbors_indices = batch['anchor_neighbors_indices']
        neighbors_output = neighbors_outputs[neighbors_indices].cuda(non_blocking=True)

        anchors_weak = output[0]
        neighbors2 = neighbors_output

        weak_anchors_prob = anchors_weak.softmax(1)
        neighbors_prob = neighbors2.softmax(2)

        # set ct1
        max_prob_0, target_0 = torch.max(weak_anchors_prob, dim=1)
        mask0 = max_prob_0 > ct1

        weak_anchors_prob = weak_anchors_prob[mask0]
        neighbors_prob = neighbors_prob[mask0]
        b, c = weak_anchors_prob.size()

        beta = torch.sum(torch.exp(-torch.norm(weak_anchors_prob.unsqueeze(1) - neighbors_prob, dim=2) ** 2).unsqueeze(
            2) * neighbors_prob, dim=1)
        beta = beta / beta.sum(1).view(-1, 1).expand_as(beta)

        # compute the tau
        q_beta_norm = torch.norm(weak_anchors_prob - beta, dim=1) ** 2

        topk = max(int(eta * b), 1)
        topk_min, _ = torch.topk(q_beta_norm, topk, largest=False) # topk
        tau = topk_min[-1] / torch.exp(torch.tensor([-1.0]).cuda())
        alpha = -torch.log(q_beta_norm / tau)

        # hardening based on alpha and mask0
        q = []
        for i in range(len(alpha)):
            if alpha[i] > 1:
                qi = weak_anchors_prob[i] ** alpha[i]
                qi = qi / qi.sum(0)
            else:
                qi = weak_anchors_prob[i]
            q.append(qi.unsqueeze(0))
        q = torch.cat(q, dim=0)

        output = [q]

        mask = None
        for i, output_i in enumerate(output):
            max_prob, target = torch.max(output_i, dim=1)
            mask = max_prob > ct # mask:cuda(0)
            predictions[i].append(torch.masked_select(torch.argmax(output_i, dim=1), mask.squeeze()))
            probs[i].append(output_i[mask])
            sum += torch.sum(mask>0)

        alphas.append(torch.masked_select(alpha, mask.squeeze()))
        targets.append(batch['target'][mask0][mask]) # target:cpu, mask0:cuda(0), mask:cuda(0), batch['target']: cpu,targets是list
        indices.append(batch['index'][mask0][mask]) # indices:cpu

        if include_neighbors:
            neighbors.append(batch['anchor_neighbors_indices'][mask0][mask] )

    predictions = [torch.cat(pred_, dim=0).cpu() for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0).cpu() for prob_ in probs]
    targets = torch.cat(targets, dim=0)
    indices = torch.cat(indices, dim=0) # list变tensor，保持cpu
    alphas = torch.cat(alphas, dim=0)


    if include_neighbors:
        neighbors = torch.cat(neighbors, dim=0)
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'neighbors': neighbors, 'alphas': alphas, 'indices': indices} for
               pred_, prob_ in zip(predictions, probs)]

    else:
        out = [{'predictions': pred_, 'probabilities': prob_, 'targets': targets, 'alphas': alphas, 'indices': indices} for pred_, prob_ in
               zip(predictions, probs)]

    if return_features:
        return out, features.cpu()
    else:
        return out, sum

@torch.no_grad()
def get_pseudo_labels(indexs, outputs):
    predictions = []
    predictions = torch.argmax(outputs, dim=1)
    predictions_order = torch.zeros_like(predictions) - 1
    predictions_order[indexs] = predictions
    return predictions_order


@torch.no_grad()
def get_confident_simples_ind(train_dataloader, model, r=0.05):
    model.eval()
    targets, features, indexs = [], [], []
    for i, batch in enumerate(train_dataloader):
        # Forward pass
        input = batch[0].cuda(non_blocking=True)
        with torch.no_grad():
            feature_ = model(input, input, forward_type='embedding')

        features.append(feature_)


    features = torch.cat(features)

    # get confi
    features = torch.cat(features)

    predictions = features.softmax(dim=1)
    sorted, ind = torch.sort(predictions, dim=0, descending=True)
    conf_ind = ind[0:int(predictions.shape[0]*r), :]

    features_conf = features[conf_ind]
    centroids = torch.sum(features_conf, dim=0)/features_conf.shape[0]    # 10*10, each row represents a centroid_i.

    conf_indexs = indexs[conf_ind].reshape(1, -1)  # get the real index. indexs are from batch samples.


    outputs = features.softmax(dim=1)
    val, _ = torch.max(outputs, dim=1)
    # conf_ind = indexs[val > torch.mean(val)]
    return conf_ind


@torch.no_grad()
def cc_evaluate(predictions, p):
    # Evaluate model based on SCAN loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        anchors = torch.arange(neighbors.size(0)).view(-1, 1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

        # Consistency loss
        similarity = torch.matmul(probs, probs.t())
        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)
        similarity = similarity[anchors, neighbors]

        pos_similarity = torch.sum(similarity)/probs.shape[0]
        consistency_loss = -pos_similarity

        # Total loss
        total_loss = - entropy_loss + consistency_loss

        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}


@torch.no_grad()
def hungarian_evaluate(subhead_index, all_predictions, class_names=None,
                       compute_purity=True, compute_confusion_matrix=True,
                       confusion_matrix_file=None):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    probs = head['probabilities'].cuda()
    num_classes = torch.unique(targets).numel()
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predictions.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predictions.cpu().numpy())

    _, preds_top5 = probs.topk(5, 1, largest=True)
    reordered_preds_top5 = torch.zeros_like(preds_top5)
    for pred_i, target_i in match:
        reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    correct_top5_binary = reordered_preds_top5.eq(targets.view(-1, 1).expand_as(reordered_preds_top5))
    top5 = float(correct_top5_binary.sum()) / float(num_elems)

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds.cpu().numpy(), targets.cpu().numpy(),
                         class_names, confusion_matrix_file, )

    return {'ACC': acc, 'ARI': ari, 'NMI': nmi, 'ACC Top-5': top5, 'hungarian_match': match}


@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res


@torch.no_grad()
def scan_evaluate(predictions):
    # Evaluate model based on SCAN loss.
    num_heads = len(predictions)
    output = []

    for head in predictions:
        # Neighbors and anchors
        probs = head['probabilities']
        neighbors = head['neighbors']
        anchors = torch.arange(neighbors.size(0)).view(-1, 1).expand_as(neighbors)

        # Entropy loss
        entropy_loss = entropy(torch.mean(probs, dim=0), input_as_probabilities=True).item()

        # Consistency loss
        similarity = torch.matmul(probs, probs.t())

        neighbors = neighbors.contiguous().view(-1)
        anchors = anchors.contiguous().view(-1)

        similarity = similarity[anchors, neighbors]
        ones = torch.ones_like(similarity)
        consistency_loss = F.binary_cross_entropy(similarity, ones).item()

        # Total loss
        total_loss = - entropy_loss + consistency_loss

        output.append({'entropy': entropy_loss, 'consistency': consistency_loss, 'total_loss': total_loss})

    total_losses = [output_['total_loss'] for output_ in output]
    lowest_loss_head = np.argmin(total_losses)
    lowest_loss = np.min(total_losses)

    return {'scan': output, 'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}

def kmeans(features, targets):

    num_elems = targets.size(0)
    num_classes = torch.unique(targets).numel()
    print('num_classes%d'%num_classes)
    kmeans = KMeans(n_clusters=num_classes, n_init=10)
    predicted = kmeans.fit_predict(features.numpy())


    # 可视化
    import matplotlib.pyplot as plt
    from sklearn import manifold
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # learning_rate="auto"加入之后结果变差
    Y = tsne.fit_transform(features.numpy())
    plt.scatter(Y[:, 0], Y[:, 1], c=predicted)
    plt.show()


    predicted = torch.from_numpy(predicted)
    predicted = predicted.cuda(targets.device)

    match = _hungarian_match(predicted, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predicted.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predicted == int(pred_i)] = int(target_i)

    # Gather performance metrics
    acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(targets.cpu().numpy(), predicted.cpu().numpy())
    ari = metrics.adjusted_rand_score(targets.cpu().numpy(), predicted.cpu().numpy())

    print({'ACC': acc, 'ARI': ari, 'NMI': nmi})

    return predicted

