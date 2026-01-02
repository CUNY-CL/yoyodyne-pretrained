"""Validation metrics.

The computation of loss is built into the models. A slight modification of
a built-in class from torchmetrics is used to compute exact match accuracy.

Adding additional metrics is relatively easy, though there are a lot of steps.
Suppose one wants to add a metric called Wham. Then one must:

* Implement `Wham(torchmetrics.Metric)` in this module.
* Add the following to the `BaseModel` in `model.py`:
    - add `wham: metrics.Wham | None` to the member type declarations
    - add `compute_wham=False` to the constructor's arguments
    - add `self.wham = metric.Wham(...) if compute_wham else None` to the
      body of the constructor
    - add the following property:

        @property
        def has_wham(self) -> bool:
            return self.wham is not None

    - add the following to the body of `_reset_metrics`:

        if self.has_wham:
            self.wham.reset()

    - add the following to the body of `_update_metrics`:

        if self.has_wham:
            self.wham.update(predictions, target)

    - add the following to the body of `_log_metrics_on_epoch_end`:

        if self.has_wham:
            self.log(
                f"{subset}_wham",
                self.wham.compute(),
                logger=True,
                on_epoch=True,
                prog_bar=True,
            )
"""

import torch
from torch import nn
import torchmetrics


class Error(Exception):
    pass


class Accuracy(torchmetrics.classification.MulticlassExactMatch):
    """Exact match string accuracy ignoring padding symbols."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def update(self, hypo: torch.Tensor, gold: torch.Tensor) -> None:
        """Accumulates accuracy sufficient statistics for a batch.

        This also performs all the necessary data validation work.

        Args:
            hypo (torch.Tensor): a tensor of hypothesis data of shape
                B x target_vocab_size x seq_len or B x seq_len.
            gold (torch.Tensor): a tensor of gold data of shape B x seq_len.

        Raises:
            Error: Hypothesis tensor is not 2d or 3d.
            Error: Gold tensor is not 2d.
            Error: Hypothesis and gold batch sizes do not match.
            Error: Hypothesis string lengths exceeds precision.
            Error: Gold string lengths exceeds precision.
        """
        if hypo.ndim < 2 or hypo.ndim > 3:
            raise Error(f"Hypothesis tensor is not 2d or 3d ({hypo.ndim})")
        if hypo.ndim == 3:
            hypo = torch.argmax(hypo, dim=1)
        if gold.ndim != 2:
            raise Error(f"Gold tensor is not 2d ({gold.ndim})")
        if hypo.size(0) != gold.size(0):
            raise Error(
                "Hypothesis and gold batch sizes do not match "
                f"({gold.size(0)} != {hypo.size(0)})"
            )
        if hypo.size(1) != gold.size(1):
            max_len = max(hypo.size(1), gold.size(1))
            if hypo.size(1) < max_len:
                hypo = nn.functional.pad(
                    hypo, (0, max_len - hypo.size(1)), value=self.ignore_index
                )
            if gold.size(1) < max_len:
                gold = nn.functional.pad(
                    gold, (0, max_len - gold.size(1)), value=self.ignore_index
                )
        super().update(hypo, gold)
