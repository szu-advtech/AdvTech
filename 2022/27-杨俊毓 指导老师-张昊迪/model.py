import torch
import torch.nn as nn
from random import sample
import numpy as np
import torch.nn.functional as F

class PiCO(nn.Module):

    def __init__(self, args, base_encoder):
        super().__init__()
        
        pretrained = args.dataset == 'cub200'
        # we allow pretraining for CUB200, or the network will not converge

        # yjuny:定义两个Encoder
        self.encoder_q = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)
        # momentum encoder
        self.encoder_k = base_encoder(num_class=args.num_class, feat_dim=args.low_dim, name=args.arch, pretrained=pretrained)

        # yjuny:encoder_k的参数不可梯度更新
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # yjuny:创建各条队列 /负样本/伪标签/ptr(队列指针)/原型向量
        self.register_buffer("queue", torch.randn(args.moco_queue, args.low_dim))
        self.register_buffer("queue_pseudo", torch.randn(args.moco_queue))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))        
        self.register_buffer("prototypes", torch.zeros(args.num_class,args.low_dim))
        self.queue = F.normalize(self.queue, dim=0)

    # yjuny:动量更新encoder_k
    @torch.no_grad()
    def _momentum_update_key_encoder(self, args):
        """
        update momentum encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * args.moco_m + param_q.data * (1. - args.moco_m)

    # yjuny:负样本入列与出列操作
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, args):
        # gather keys before updating queue
        # 更新队列前聚集所有keys和labels
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]

        # 判断队列的大小是否为batch大小的整数倍
        ptr = int(self.queue_ptr)
        assert args.moco_queue % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        # yjuny:以指针的方式替换keys
        self.queue[ptr:ptr + batch_size, :] = keys
        self.queue_pseudo[ptr:ptr + batch_size] = labels
        ptr = (ptr + batch_size) % args.moco_queue  # 移除指针

        self.queue_ptr[0] = ptr

    # yjuny:DDP分布式多卡计算(乱序版) 打乱batch内样本索引
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index 随机打乱样本索引
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus 广播至所有的GPU
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring 恢复索引
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu 当前GPU的乱序索引
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    # yjuny:恢复样本索引
    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)

        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    # yjuny:模型前向传播
    def forward(self, img_q, im_k=None, partial_Y=None, args=None, eval_only=False):
        
        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        # for testing

        # yjuny:通过伪标签的概率分布softmax得到 样本伪标签pseudo_labels_b
        predicted_scores = torch.softmax(output, dim=1) * partial_Y
        max_scores, pseudo_labels_b = torch.max(predicted_scores, dim=1)
        # using partial labels to filter out negative labels
        # print('pseudo_labels_b:', pseudo_labels_b)

        # compute protoypical logits
        # yjuny:查看伪标签向量和哪个原型向量最接近
        prototypes = self.prototypes.clone().detach()
        logits_prot = torch.mm(q, prototypes.t())
        score_prot = torch.softmax(logits_prot, dim=1)

        # update momentum prototypes with pseudo labels
        # yjuny:动量更新原型标签
        for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels_b)):
            self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
        # normalize prototypes    
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
        
        # compute key features 计算 key 特征
        with torch.no_grad():  # no gradient 
            self._momentum_update_key_encoder(args)  # update the momentum encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        # to calculate SupCon Loss using pseudo_labels
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, pseudo_labels_b, args)

        return output, features, pseudo_labels, score_prot, pseudo_labels_b


class PiCO_PLUS(PiCO):
    '''yjuny:PiCO+是PiCO的强大扩展，能够缓解嘈杂的部分标签学习问题'''
    def __init__(self, args, base_encoder):
        super().__init__(args, base_encoder)
        # yjuny:相关性队列
        self.register_buffer("queue_rel", torch.zeros(args.moco_queue, dtype=torch.bool))

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels, is_rel, args):
        super()._dequeue_and_enqueue(keys, labels, args)
        # change:yjuny:聚类的时候选择比较靠谱的
        is_rel = concat_all_gather(is_rel)
        batch_size = is_rel.shape[0]
        ptr = int(self.queue_ptr)
        self.queue_rel[ptr:ptr + batch_size] = is_rel
        # update queue_rel

    def forward(self, img_q, im_k=None, Y_ori=None, Y_cor=None, is_rel=None, args=None, eval_only=False, ):

        output, q = self.encoder_q(img_q)
        if eval_only:
            return output
        # for testing

        batch_weight = is_rel.float()
        with torch.no_grad():  # no gradient 

            predicetd_scores = torch.softmax(output, dim=1)
            # yjuny:一个batch里面的所有类别
            _, within_max_cls = torch.max(predicetd_scores * Y_ori, dim=1)
            _, all_max_cls = torch.max(predicetd_scores, dim=1)
            pseudo_labels_b = batch_weight * within_max_cls + (1 - batch_weight) * all_max_cls
            pseudo_labels_b = within_max_cls
            pseudo_labels_b = pseudo_labels_b.long()
            # for clean data, using partial labels to filter out negative labels
            # for noisy data, we enable a full set pseudo-label selection

            # compute protoypical logits
            prototypes = self.prototypes.clone().detach()
            logits_prot = torch.mm(q, prototypes.t())
            score_prot = torch.softmax(logits_prot, dim=1)
            # prototypes follows the same

            # change:yjuny:在这里，我们使用到分类器预测的原始集合原型内的距离来检测候选标签集是否有噪声。
            # change:yjuny:如果实例远离分类器预测的原型，它可能违反对比学习的聚类趋势，因此我们将其视为有噪声
            # change:yjuny:is_rel正是我们用来判断哪些是噪音样本的标记向量(可靠程度)
            _, within_max_cls_ori = torch.max(predicetd_scores * Y_ori, dim=1)
            distance_prot = - (q * prototypes[within_max_cls_ori]).sum(dim=1)
            # Here we use the distances to those within the original set prototype of classifier prediction
            #       to detect whether a candidate label set is noisy
            # if the instance is far away from the classifier-predicted prototype,
            #       it may violate the clustering tendency of the contrastive learning
            #       and hence we regard it as noisy

            # update momentum prototypes with pseudo labels
            # yjuny:动量更新原型标签
            for feat, label in zip(concat_all_gather(q[is_rel]), concat_all_gather(pseudo_labels_b[is_rel])):
            # for feat, label in zip(concat_all_gather(q), concat_all_gather(pseudo_labels_b)):
                self.prototypes[label] = self.prototypes[label]*args.proto_m + (1-args.proto_m)*feat
            # normalize prototypes
            self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
            # print:
            # print('prototypes:', self.prototypes, self.prototypes.shape)
            # print('queue_pseudo:', self.queue_pseudo, self.queue_pseudo.shape)
            
            # compute key features 
            self._momentum_update_key_encoder(args)  # update the momentum encoder
            # shuffle for making use of BN
            im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)
            _, k = self.encoder_k(im_k)
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

        features = torch.cat((q, k, self.queue.clone().detach()), dim=0)
        # print:
        # print('self.queue:', self.queue, self.queue.shape)
        # print('pseudo_labels_before:', pseudo_labels_b, pseudo_labels_b.shape)
        pseudo_labels = torch.cat((pseudo_labels_b, pseudo_labels_b, self.queue_pseudo.clone().detach()), dim=0)
        # print:
        # print('pseudo_labels:', pseudo_labels, pseudo_labels.shape)
        is_rel_queue = torch.cat((is_rel, is_rel, self.queue_rel.clone().detach()), dim=0)
        # to calculate SupCon Loss using pseudo_labels and partial target
        
        # dequeue and enqueue
        self._dequeue_and_enqueue(k, pseudo_labels_b, is_rel, args)

        return output, features, pseudo_labels, score_prot, distance_prot, is_rel_queue, within_max_cls

# utils
# 聚合函数 PiCO使用多GPU进行分布式训练，这使得一个batch的样本被拆分至多个GPU上，因此在对字典编码器更新时需要汇聚所有GPU上的样本。
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
