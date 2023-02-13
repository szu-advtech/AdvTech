import logging
import os

import numpy as np
import pytorch_lightning as pl
import torchmetrics
import torch
import transformers
from transformers import AutoModel
import torch.nn.functional as F
import torch.nn as nn
import sys

sys.path.insert(0, '..')

import utils
from metrics import metrics

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO)

logger = logging.getLogger()


class ProtoModule(pl.LightningModule):

    def __init__(self,
                 pretrained_model,
                 num_classes,
                 label_order_path,
                 use_sigmoid=False,
                 use_cuda=True,
                 lr_prototypes=5e-2,
                 lr_features=2e-6,
                 lr_others=2e-2,
                 num_training_steps=5000,
                 num_warmup_steps=1000,
                 loss='BCE',
                 save_dir='output',
                 use_attention=True,
                 dot_product=False,
                 normalize=None,
                 final_layer=False,
                 reduce_hidden_size=None,
                 use_prototype_loss=False,
                 prototype_vector_path=None,
                 attention_vector_path=None,
                 eval_buckets=None,
                 seed=7,
                 contrastive_loss = True
                 ):

        super().__init__()
        self.label_order_path = label_order_path

        self.loss = loss
        self.normalize = normalize
        self.lr_features = lr_features
        self.lr_prototypes = lr_prototypes
        self.lr_others = lr_others
        self.use_sigmoid = use_sigmoid
        self.use_cuda = use_cuda

        self.use_attention = use_attention
        self.dot_product = dot_product

        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.save_dir = save_dir
        self.num_classes = num_classes

        self.final_layer = final_layer
        self.use_prototype_loss = use_prototype_loss
        self.prototype_vector_path = prototype_vector_path
        self.eval_buckets = eval_buckets

        #对比学习模块
        self.contrastive_loss = contrastive_loss
        # 模型架构设置 #

        pl.utilities.seed.seed_everything(seed=seed)

        # 定义pairwist_dist模型的度量
        self.pairwise_dist = nn.PairwiseDistance(p=2)

        # huggingface加载预训练模型PubMedBert
        self.bert = AutoModel.from_pretrained(pretrained_model)

        # 冻结 BERT 的层 if lr_features == 0
        if lr_features == 0:
            for param in self.bert.parameters():
                param.requires_grad = False

        # 为了减少内存占用需要进行维度缩减
        self.hidden_size = self.bert.config.hidden_size
        self.reduce_hidden_size = reduce_hidden_size is not None
        if self.reduce_hidden_size:
            self.reduce_hidden_size = True
            self.bert_hidden_size = self.bert.config.hidden_size
            self.hidden_size = reduce_hidden_size

            # 线性层初始化
            # reset the seed to make sure linear layer is the same as in preprocessing
            pl.utilities.seed.seed_everything(seed=seed)
            self.linear = nn.Linear(self.bert_hidden_size, self.hidden_size)

        # 如果有现成的原型向量可以直接载入
        if prototype_vector_path is not None:
            prototype_vectors, self.num_prototypes_per_class = self.load_prototype_vectors(prototype_vector_path)
        else:
            prototype_vectors = torch.rand((self.num_classes, self.hidden_size))
            self.num_prototypes_per_class = torch.ones(self.num_classes)
        self.prototype_vectors = nn.Parameter(prototype_vectors, requires_grad=True)

        self.prototype_to_class_map = self.build_prototype_to_class_mapping(self.num_prototypes_per_class)
        self.num_prototypes = self.prototype_to_class_map.shape[0]

        # 如果有现成的attention向量可以直接载入
        if attention_vector_path is not None:
            attention_vectors = self.load_attention_vectors(attention_vector_path)
        else:
            attention_vectors = torch.rand((self.num_classes, self.hidden_size))
        self.attention_vectors = nn.Parameter(attention_vectors, requires_grad=True)

        if self.final_layer:
            self.final_linear = self.build_final_layer()

        # EVALUATION SETUP #

        # 基础的metrics
        self.train_metrics = self.setup_metrics()

        # 在测试阶段加入额外的metrics
        self.all_metrics = {**self.train_metrics, **self.setup_extensive_metrics()}

        self.save_hyperparameters()
        logger.info("Finished init.")

    def build_final_layer(self):
        prototype_identity_matrix = torch.zeros(self.num_prototypes, self.num_classes)

        for j in range(len(prototype_identity_matrix)):
            prototype_identity_matrix[j, self.prototype_to_class_map[j]] = 1.0 / self.num_prototypes_per_class[
                self.prototype_to_class_map[j]]

        if self.use_cuda:
            prototype_identity_matrix = prototype_identity_matrix.cuda()

        return nn.Parameter(prototype_identity_matrix.double(), requires_grad=True)

    def load_prototype_vectors(self, prototypes_per_class_path):
        prototypes_per_class = torch.load(prototypes_per_class_path)

        # 存储每个类的prototype的个数
        num_prototypes_per_class = torch.tensor([len(prototypes_per_class[key]) for key in prototypes_per_class])

        with open(self.label_order_path) as label_order_file:
            ordered_labels = label_order_file.read().split(" ")

        # 获得存储的prototype的dimension的数量
        vector_dim = len(list(prototypes_per_class.values())[0][0])

        stacked_prototypes_per_class = [
            prototypes_per_class[label] if label in prototypes_per_class else [np.random.rand(vector_dim)]
            for label in ordered_labels]

        prototype_matrix = torch.tensor([val for sublist in stacked_prototypes_per_class for val in sublist])

        return prototype_matrix, num_prototypes_per_class

    def build_prototype_to_class_mapping(self, num_prototypes_per_class):
        return torch.arange(num_prototypes_per_class.shape[0]).repeat_interleave(num_prototypes_per_class.long(),
                                                                                 dim=0)

    def load_attention_vectors(self, attention_vectors_path):
        attention_vectors = torch.load(attention_vectors_path, map_location=self.device)

        return attention_vectors

    def setup_metrics(self):
        self.f1 = torchmetrics.F1(threshold=0.269)
        self.auroc_micro = metrics.FilteredAUROC(num_classes=self.num_classes, compute_on_step=False, average="micro")
        self.auroc_macro = metrics.FilteredAUROC(num_classes=self.num_classes, compute_on_step=False, average="macro")

        return {"auroc_micro": self.auroc_micro,
                "auroc_macro": self.auroc_macro,
                "f1": self.f1}

    def setup_extensive_metrics(self):
        self.pr_curve = metrics.PR_AUC(num_classes=self.num_classes)

        extensive_metrics = {"pr_curve": self.pr_curve}

        if self.eval_buckets:
            buckets = self.eval_buckets

            self.prcurve_0 = metrics.PR_AUCPerBucket(bucket=buckets["<5"],
                                                     num_classes=self.num_classes,
                                                     compute_on_step=False)

            self.prcurve_1 = metrics.PR_AUCPerBucket(bucket=buckets["5-10"],
                                                     num_classes=self.num_classes,
                                                     compute_on_step=False)

            self.prcurve_2 = metrics.PR_AUCPerBucket(bucket=buckets["11-50"],
                                                     num_classes=self.num_classes,
                                                     compute_on_step=False)

            self.prcurve_3 = metrics.PR_AUCPerBucket(bucket=buckets["51-100"],
                                                     num_classes=self.num_classes,
                                                     compute_on_step=False)

            self.prcurve_4 = metrics.PR_AUCPerBucket(bucket=buckets["101-1K"],
                                                     num_classes=self.num_classes,
                                                     compute_on_step=False)

            self.prcurve_5 = metrics.PR_AUCPerBucket(bucket=buckets[">1K"],
                                                     num_classes=self.num_classes,
                                                     compute_on_step=False)

            self.auroc_macro_0 = metrics.FilteredAUROCPerBucket(bucket=buckets["<5"],
                                                                num_classes=self.num_classes,
                                                                compute_on_step=False,
                                                                average="macro")
            self.auroc_macro_1 = metrics.FilteredAUROCPerBucket(bucket=buckets["5-10"],
                                                                num_classes=self.num_classes,
                                                                compute_on_step=False,
                                                                average="macro")
            self.auroc_macro_2 = metrics.FilteredAUROCPerBucket(bucket=buckets["11-50"],
                                                                num_classes=self.num_classes,
                                                                compute_on_step=False,
                                                                average="macro")
            self.auroc_macro_3 = metrics.FilteredAUROCPerBucket(bucket=buckets["51-100"],
                                                                num_classes=self.num_classes,
                                                                compute_on_step=False,
                                                                average="macro")
            self.auroc_macro_4 = metrics.FilteredAUROCPerBucket(bucket=buckets["101-1K"],
                                                                num_classes=self.num_classes,
                                                                compute_on_step=False,
                                                                average="macro")
            self.auroc_macro_5 = metrics.FilteredAUROCPerBucket(bucket=buckets[">1K"],
                                                                num_classes=self.num_classes,
                                                                compute_on_step=False,
                                                                average="macro")

            bucket_metrics = {"pr_curve_0": self.prcurve_0,
                              "pr_curve_1": self.prcurve_1,
                              "pr_curve_2": self.prcurve_2,
                              "pr_curve_3": self.prcurve_3,
                              "pr_curve_4": self.prcurve_4,
                              "pr_curve_5": self.prcurve_5,
                              "auroc_macro_0": self.auroc_macro_0,
                              "auroc_macro_1": self.auroc_macro_1,
                              "auroc_macro_2": self.auroc_macro_2,
                              "auroc_macro_3": self.auroc_macro_3,
                              "auroc_macro_4": self.auroc_macro_4,
                              "auroc_macro_5": self.auroc_macro_5}

            extensive_metrics = {**extensive_metrics, **bucket_metrics}

        return extensive_metrics

    def configure_optimizers(self):
        joint_optimizer_specs = [{'params': self.prototype_vectors, 'lr': self.lr_prototypes},
                                 {'params': self.attention_vectors, 'lr': self.lr_others},
                                 {'params': self.bert.parameters(), 'lr': self.lr_features}]

        if self.final_layer:
            joint_optimizer_specs.append({'params': self.final_linear, 'lr': self.lr_prototypes})

        if self.reduce_hidden_size:
            joint_optimizer_specs.append({'params': self.linear.parameters(), 'lr': self.lr_others})

        optimizer = torch.optim.AdamW(joint_optimizer_specs)

        lr_scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )

        return [optimizer], [lr_scheduler]

    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)

    def training_step(self, batch, batch_idx):
        targets = torch.tensor(batch['targets'], device=self.device)

        if self.use_prototype_loss:
            if batch_idx == 0:
                self.prototype_loss = self.calculate_prototype_loss()
            self.log('prototype_loss', self.prototype_loss, on_epoch=True)

        logits, _ = self(batch)

        if self.loss == "BCE":
            train_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target=targets.float())
        else:
            train_loss = torch.nn.MultiLabelSoftMarginLoss()(input=torch.sigmoid(logits), target=targets)

        self.log('train_loss', train_loss, on_epoch=True)

        if self.use_prototype_loss:
            total_loss = train_loss + self.prototype_loss
        else:
            total_loss = train_loss

        return total_loss
    def Contrastive_Loss(self,targets,bert_output,Tempreture=10):
        beta = torch.einsum('bn,cn->bc',targets.float(),targets.float()) # bs*bs
        # 去掉
        
        beta = beta-torch.diag_embed(torch.diag(beta))
        beta = F.normalize(beta,p=1,dim=1,eps=1e-8)
        disance_matrix = -torch.einsum('btn,ctn->bc',bert_output,bert_output)
        disance_matrix = disance_matrix - torch.diag_embed(torch.diag(disance_matrix))
        # bs * bs 
        # 温度系数
        disance_matrix /= Tempreture

        # labels = torch.zeros(disance_matrix.shape[0],dtype=torch.long).cuda()
        # loss = F.cross_entropy(disance_matrix,labels)
        
        loss_matrix = -beta*F.log_softmax(disance_matrix,dim=1) 
        # mask = beta.nonzero
        # loss_matrix = loss_matrix*mask
        loss = torch.sum(loss_matrix)
        
        return loss
    
    def forward(self, batch):
        attention_mask = batch["attention_masks"]
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]

        if attention_mask.device != self.device:
            attention_mask = attention_mask.to(self.device)
            input_ids = input_ids.to(self.device)
            token_type_ids = token_type_ids.to(self.device)

        bert_output = self.bert(input_ids=input_ids,
                                attention_mask=attention_mask,
                                token_type_ids=token_type_ids)

        bert_vectors = bert_output.last_hidden_state

        if self.reduce_hidden_size:
            # apply linear layer to reduce token vector dimension
            token_vectors = self.linear(bert_vectors)
        else:
            token_vectors = bert_vectors

        if self.normalize is not None:
            token_vectors = nn.functional.normalize(token_vectors, p=2, dim=self.normalize)

        metadata = None
        if self.use_attention:

            attention_mask_from_tokens = utils.attention_mask_from_tokens(attention_mask, batch["tokens"])

            weighted_samples_per_class, attention_per_token_and_class = self.calculate_token_class_attention(
                token_vectors,
                self.attention_vectors,
                mask=attention_mask_from_tokens)

            if self.normalize is not None:
                weighted_samples_per_class = nn.functional.normalize(weighted_samples_per_class, p=2,
                                                                     dim=self.normalize)

            if self.use_cuda:
                weighted_samples_per_class = weighted_samples_per_class.cuda()
                self.num_prototypes_per_class = self.num_prototypes_per_class.cuda()

            weighted_samples_per_prototype = weighted_samples_per_class.repeat_interleave(
                self.num_prototypes_per_class
                    .long(), dim=1)

            if self.dot_product:
                score_per_prototype = torch.einsum('bs,abs->ab', self.prototype_vectors,
                                                   weighted_samples_per_prototype)
            else:
                score_per_prototype = -self.pairwise_dist(self.prototype_vectors.T,
                                                          weighted_samples_per_prototype.permute(0, 2, 1))

            metadata = attention_per_token_and_class, weighted_samples_per_prototype

        else:
            score_per_prototype = -torch.cdist(token_vectors.mean(dim=1), self.prototype_vectors)

        logits = self.get_logits_per_class(score_per_prototype)

        return logits, metadata

    def calculate_token_class_attention(self, batch_samples, class_attention_vectors, mask=None):
        if class_attention_vectors.device != batch_samples.device:
            class_attention_vectors = class_attention_vectors.to(batch_samples.device)

        score_per_token_and_class = torch.einsum('ikj,mj->imk', batch_samples, class_attention_vectors)

        if mask is not None:
            expanded_mask = mask.unsqueeze(dim=1).expand(mask.size(0), class_attention_vectors.size(0), mask.size(1))

            expanded_mask = F.pad(input=expanded_mask,
                                  pad=(0, score_per_token_and_class.shape[2] - expanded_mask.shape[2]),
                                  mode='constant', value=0)

            score_per_token_and_class = score_per_token_and_class.masked_fill(
                (expanded_mask == 0),
                float('-inf'))

        if self.use_sigmoid:
            attention_per_token_and_class = torch.sigmoid(score_per_token_and_class) / \
                                            score_per_token_and_class.shape[2]
        else:
            attention_per_token_and_class = F.softmax(score_per_token_and_class, dim=2)

        class_weighted_tokens = torch.einsum('ikjm,ikj->ikjm',
                                             batch_samples.unsqueeze(dim=1).expand(batch_samples.size(0),
                                                                                   self.num_classes,
                                                                                   batch_samples.size(1),
                                                                                   batch_samples.size(2)),
                                             attention_per_token_and_class)

        weighted_samples_per_class = class_weighted_tokens.sum(dim=2)

        return weighted_samples_per_class, attention_per_token_and_class

    def get_logits_per_class(self, score_per_prototype):
        if self.final_layer:
            if score_per_prototype.device != self.final_linear.device:
                score_per_prototype = score_per_prototype.to(self.final_linear.device)

            return torch.matmul(score_per_prototype, self.final_linear)

        else:
            batch_size = score_per_prototype.shape[0]

            fill_vector = torch.full((batch_size, self.num_classes, self.num_prototypes), fill_value=float("-inf"),
                                     dtype=score_per_prototype.dtype)
            if self.use_cuda:
                fill_vector = fill_vector.cuda()
                self.prototype_to_class_map = self.prototype_to_class_map.cuda()

            group_logits_by_class = fill_vector.scatter_(1,
                                                         self.prototype_to_class_map.unsqueeze(0).repeat(batch_size,
                                                                                                         1).unsqueeze(
                                                             1),
                                                         score_per_prototype.unsqueeze(1))

            max_logits_per_class = torch.max(group_logits_by_class, dim=2).values
            return max_logits_per_class

    def calculate_prototype_loss(self):
        prototype_loss = 100 / torch.tensor([torch.cdist(
            self.prototype_vectors[(self.prototype_to_class_map == i).nonzero().flatten()][:1],
            self.prototype_vectors[(self.prototype_to_class_map == i).nonzero().flatten()][1:]).min() for i in
                                             range(self.num_classes) if
                                             len((self.prototype_to_class_map == i).nonzero()) > 1]).sum()
        return prototype_loss

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            targets = torch.tensor(batch['targets'], device=self.device)

            logits, _ = self(batch)

            for metric_name in self.train_metrics:
                metric = self.train_metrics[metric_name]
                metric(torch.sigmoid(logits), targets)

    def validation_epoch_end(self, outputs) -> None:
        for metric_name in self.train_metrics:
            metric = self.train_metrics[metric_name]
            self.log(f"val/{metric_name}", metric.compute())
            metric.reset()

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            targets = torch.tensor(batch['targets'], device=self.device)

            logits, _ = self(batch)
            preds = torch.sigmoid(logits)

            for metric_name in self.all_metrics:
                metric = self.all_metrics[metric_name]
                metric(preds, targets)

        return preds, targets

    def test_epoch_end(self, outputs) -> None:
        log_dir = self.logger.log_dir
        for metric_name in self.all_metrics:
            metric = self.all_metrics[metric_name]
            value = metric.compute()
            self.log(f"test/{metric_name}", value)

            with open(os.path.join(log_dir, 'test_metrics.txt'), 'a') as metrics_file:
                metrics_file.write(f"{metric_name}: {value}\n")

            metric.reset()

        predictions = torch.cat([out[0] for out in outputs])

        targets = torch.cat([out[1] for out in outputs])

        pr_auc = metrics.calculate_pr_auc(prediction=predictions, target=targets, num_classes=self.num_classes,
                                          device=self.device)

        with open(os.path.join(self.logger.log_dir, 'PR_AUC_score.txt'), 'w') as metrics_file:
            metrics_file.write(f"PR AUC: {pr_auc.cpu().numpy()}\n")
