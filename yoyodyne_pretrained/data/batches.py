"""Batch objects.

Adapted from UDTube:

    https://github.com/CUNY-CL/udtube/blob/master/udtube/data/batches.py
"""

import torch
from torch import nn


class Batch(nn.Module):
    """String data batch.

    Args:
        source: tokenized source.
        source_mask: source mask.
        target: optional tokenized target.
        source_mask: optional target mask.
    """

    source: torch.Tensor
    source_mask: torch.Tensor
    target: torch.Tensor | None
    target_mask: torch.Tensor | None

    def __init__(self, source, source_mask, target=None, target_mask=None):
        super().__init__()
        self.register_buffer("source", source)
        self.register_buffer("source_mask", source_mask)
        self.register_buffer("target", target)
        self.register_buffer("target_mask", target_mask)

    def __len__(self) -> int:
        return self.source.size(0)
