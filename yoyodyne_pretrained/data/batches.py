"""Batch objects.

Adapted from UDTube:

    https://github.com/CUNY-CL/udtube/blob/master/udtube/data/batches.py
"""

import torch
from torch import nn


class Tokenized(nn.Module):
    """Stores input IDs and attention mask in a module."""

    ids: torch.Tensor
    mask: torch.Tensor

    def __init__(self, ids, mask):
        super().__init__()
        self.register_buffer("ids", ids)
        self.register_buffer("mask", mask)


class Batch(nn.Module):
    """String data batch.

    Args:
        source: tokenized source.
        target: optional tokenized target.
    """

    source: Tokenized
    target: Tokenized | None

    def __init__(self, source, target=None):
        super().__init__()
        self.register_buffer("source", source)
        self.register_buffer("target", target)

    def __len__(self) -> int:
        return self.source.ids.size(0)
