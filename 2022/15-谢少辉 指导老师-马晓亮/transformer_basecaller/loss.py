import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import torch.backends.cudnn
from dataclasses import dataclass, field
from typing import List, Optional
from transformer_basecaller import SACallModel


@dataclass
class CTCCriterionConfig:
    zero_infinity: bool = field(
        default=False,
        metadata={"help": "zero inf loss when source length <= target length"},
    )
    blank_idx: int = field(
        default=0,
        metadata={"help": "blank token index"}
    )
    reduction: str = field(
        default="sum",
        metadata={"help": "reduction method of ctc loss"}
    )


class CTCCriterion(_Loss):
    def __init__(self, cfg: CTCCriterionConfig):
        super().__init__()
        self.zero_infinity = cfg.zero_infinity
        self.blank_idx = cfg.blank_idx
        self.reduction = cfg.reduction

    def forward(self, network_output, targets, target_lengths):
        log_probs = SACallModel.get_normalized_probs(network_output, log_probs=True).contiguous() # T x B x C
        input_lengths = log_probs.new_full(
            (log_probs.size(1),), log_probs.size(0), dtype=torch.long
        )
        with torch.backends.cudnn.flags(enabled=False):
            loss = F.ctc_loss(
                log_probs,
                targets,
                input_lengths,
                target_lengths,
                blank=self.blank_idx,
                reduction=self.reduction,
                zero_infinity=self.zero_infinity
            )

        logging_output = {
            "loss": loss.item(),
            "sample_size": targets.size(0)
        }

        return loss, targets.size(0), logging_output
