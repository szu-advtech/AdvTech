from typing import Optional, Any, Callable, List

import torch
import torchmetrics

from torchmetrics.metric import Metric
from torchmetrics import AUROC, PrecisionRecallCurve
from torchmetrics.functional.classification.auroc import _auroc_compute
from torchmetrics.utilities.data import dim_zero_cat
import logging
import numpy as np


class PR_AUC(Metric):
    def __init__(self, num_classes, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
        self.add_state("prauc", default=[], dist_reduce_fx='cat')
        self.pr_curve = PrecisionRecallCurve(num_classes=num_classes).to(self.device)
        self.auc = torchmetrics.AUC().to(self.device)

    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        precision, recall, thresholds = self.pr_curve(prediction, target)
        auc_values = [self.auc(r, p) for r, p in zip(recall, precision)]

        pr_auc = torch.mean(torch.tensor([v for v in auc_values if not v.isnan()])).to(self.device)
        self.prauc += [pr_auc.detach()]

    def compute(self):
        return torch.mean(self.prauc.detach())


class PR_AUCPerBucket(PR_AUC):
    def __init__(self, num_classes, bucket, compute_on_step=False, dist_sync_on_step=False):
        super().__init__(num_classes=len(bucket), compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
        self.bucket = set(bucket)
        self.num_classes = num_classes

    def update(self, prediction: torch.Tensor, target: torch.Tensor):

        mask = np.zeros((self.num_classes), dtype=bool)
        for c in range(self.num_classes):
            if c in self.bucket:
                mask[c] = True
        filtered_target = target[:, mask]
        filtered_preds = prediction[:, mask]

        if len((filtered_target > 0).nonzero()) > 0:
            precision, recall, thresholds = self.pr_curve(filtered_preds, filtered_target)
            auc_values = [self.auc(r, p) for r, p in zip(recall, precision)]

            pr_auc = torch.mean(torch.tensor([v for v in auc_values if not v.isnan()])).to(self.device)
            self.prauc += [pr_auc.detach()]


def calculate_pr_auc(prediction: torch.Tensor, target: torch.Tensor, num_classes, device):
    pr_curve = PrecisionRecallCurve(num_classes=num_classes).to(device)
    auc = torchmetrics.AUC().to(device)

    precision, recall, thresholds = pr_curve(prediction, target)
    auc_values = [auc(r, p) for r, p in zip(recall, precision)]

    pr_auc = torch.mean(torch.tensor([v for v in auc_values if not v.isnan()])).to(device)
    return pr_auc.detach()


class FilteredAUROC(AUROC):
    def compute(self) -> torch.Tensor:

        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        mask = np.ones((self.num_classes), dtype=bool)
        for c in range(self.num_classes):
            if torch.max(target[:, c]) == 0:
                mask[c] = False
        filtered_target = target[:, mask]
        filtered_preds = preds[:, mask]

        num_filtered_cols = np.count_nonzero(mask == False)
        logging.info(f"{num_filtered_cols} columns not considered for ROC AUC calculation!")

        return _auroc_compute(
            filtered_preds,
            filtered_target,
            self.mode,
            self.num_classes - num_filtered_cols,
            self.pos_label,
            self.average,
            self.max_fpr,
        )


class FilteredAUROCPerBucket(AUROC):
    def __init__(
            self,
            bucket: List[int],
            num_classes: Optional[int] = None,
            pos_label: Optional[int] = None,
            average: Optional[str] = "macro",
            max_fpr: Optional[float] = None,
            compute_on_step: bool = True,
            dist_sync_on_step: bool = False,
            process_group: Optional[Any] = None,
            dist_sync_fn: Callable = None
    ):
        super().__init__(num_classes, pos_label, average, max_fpr, compute_on_step, dist_sync_on_step, process_group,
                         dist_sync_fn)
        self.bucket = set(bucket)

    def compute(self) -> torch.Tensor:

        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)

        mask = np.zeros((self.num_classes), dtype=bool)
        for c in range(self.num_classes):
            if torch.max(target[:, c]) > 0 and c in self.bucket:
                mask[c] = True
        filtered_target = target[:, mask]
        filtered_preds = preds[:, mask]

        num_filtered_cols = np.count_nonzero(mask == False)
        logging.info(f"{num_filtered_cols} columns not considered for ROC AUC calculation!")

        return _auroc_compute(
            filtered_preds,
            filtered_target,
            self.mode,
            self.num_classes - num_filtered_cols,
            self.pos_label,
            self.average,
            self.max_fpr,
        )
