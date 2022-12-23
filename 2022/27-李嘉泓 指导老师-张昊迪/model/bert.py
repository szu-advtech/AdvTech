import os

import pytorch_lightning as pl
import torchmetrics
import torch
import transformers
from transformers import BertForSequenceClassification
import torch.nn.functional as F
import sys
import torch.nn as nn

sys.path.insert(0, '..')
from metrics import metrics


class BertModule(pl.LightningModule):
    def __init__(self,
                 pretrained_model,
                 num_classes,
                 lr_features=5e-6,
                 lr_others=1e-3,
                 num_training_steps=5000,
                 num_warmup_steps=1000,
                 save_dir='output',
                 eval_buckets=None,
                 reduce_hidden_size=None,
                 use_attention=False,
                 seed=7):
        super().__init__()

        self.lr_features = lr_features
        self.num_training_steps = num_training_steps
        self.num_warmup_steps = num_warmup_steps
        self.save_dir = save_dir
        self.num_classes = num_classes
        self.use_attention = use_attention
        self.lr_others = lr_others
        self.eval_buckets = eval_buckets

        self.bert = BertForSequenceClassification.from_pretrained(pretrained_model, num_labels=self.num_classes)

        # define hidden size
        self.hidden_size = self.bert.config.hidden_size
        self.reduce_hidden_size = reduce_hidden_size is not None
        if self.reduce_hidden_size:
            self.reduce_hidden_size = True
            self.bert_hidden_size = self.bert.config.hidden_size
            self.hidden_size = reduce_hidden_size

            # initialize linear layer for dim reduction
            # reset the seed to make sure linear layer is the same as in preprocessing
            pl.utilities.seed.seed_everything(seed=seed)
            self.linear = nn.Linear(self.bert_hidden_size, self.hidden_size)

        # setup metrics
        self.train_metrics = self.setup_metrics()

        # initialise metrics for evaluation on test set
        self.all_metrics = {**self.train_metrics, **self.setup_extensive_metrics()}

        if self.use_attention:
            # initialize prototype attention vectors randomly
            self.prototype_att_vectors = nn.Parameter(
                torch.rand((self.num_classes, self.hidden_size)), requires_grad=True)

            self.classifier = nn.Parameter(
                torch.rand((self.num_classes, self.hidden_size)), requires_grad=True)

        self.save_hyperparameters(ignore=["eval_buckets"])

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
        joint_optimizer_specs = [{'params': self.bert.parameters(), 'lr': self.lr_features}]

        if self.use_attention:
            joint_optimizer_specs.append({'params': self.prototype_att_vectors, 'lr': self.lr_others})
            joint_optimizer_specs.append({'params': self.classifier, 'lr': self.lr_others})

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

        logits = self(batch)

        total_loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, target=targets.float())

        self.log('train_loss', total_loss.detach(), on_epoch=True)

        return total_loss

    def forward(self, batch):
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_masks"].to(self.device)
        token_type_ids = batch["token_type_ids"].to(self.device)

        if self.use_attention:
            # add attention layer after bert's feature representation layer and
            bert_output = self.bert.bert(input_ids=input_ids,
                                         attention_mask=attention_mask,
                                         token_type_ids=token_type_ids)

            if self.reduce_hidden_size:
                # apply linear layer to reduce token vector dimension
                token_vectors = self.linear(bert_output.last_hidden_state)
            else:
                token_vectors = bert_output.last_hidden_state

            weighted_samples_per_prototype, attention_per_token_and_prototype = self.calculate_token_class_attention(
                token_vectors,
                self.prototype_att_vectors,
                mask=attention_mask)

            pooled_output = self.bert.dropout(weighted_samples_per_prototype)

            logits = torch.einsum('jik,ik->ji', pooled_output, self.classifier)

        else:
            bert_output = self.bert(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    token_type_ids=token_type_ids)
            logits = bert_output["logits"]

        return logits

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

        attention_per_token_and_class = F.softmax(score_per_token_and_class, dim=2)

        class_weighted_tokens = torch.einsum('ikjm,ikj->ikjm',
                                             batch_samples.unsqueeze(dim=1).expand(batch_samples.size(0),
                                                                                   self.num_classes,
                                                                                   batch_samples.size(1),
                                                                                   batch_samples.size(2)),
                                             attention_per_token_and_class)

        weighted_samples_per_class = class_weighted_tokens.sum(dim=2)

        return weighted_samples_per_class, attention_per_token_and_class

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            targets = torch.tensor(batch['targets'], device=self.device)

            logits = self(batch)

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

            logits = self(batch)
            preds = torch.sigmoid(logits)

            for metric_name in self.all_metrics:
                metric = self.all_metrics[metric_name]
                metric(preds, targets)

        return preds, targets

    def test_epoch_end(self, outputs) -> None:
        for metric_name in self.all_metrics:
            metric = self.all_metrics[metric_name]
            value = metric.compute()
            self.log(f"test/{metric_name}", value)

            with open(os.path.join(self.logger.log_dir, 'test_metrics.txt'), 'a') as metrics_file:
                metrics_file.write(f"{metric_name}: {value}\n")

            metric.reset()

        predictions = torch.cat([out[0] for out in outputs])

        targets = torch.cat([out[1] for out in outputs])

        pr_auc = metrics.calculate_pr_auc(prediction=predictions, target=targets, num_classes=self.num_classes,
                                          device=self.device)

        with open(os.path.join(self.logger.log_dir, 'PR_AUC_score.txt'), 'w') as metrics_file:
            metrics_file.write(f"PR AUC: {pr_auc.cpu().numpy()}")
